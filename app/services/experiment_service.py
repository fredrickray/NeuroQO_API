"""
Experiment Service - A/B testing for query optimizations.
"""
import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, TYPE_CHECKING, cast
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import numpy as np
from scipy import stats

from app.models.experiment import Experiment, ExperimentMetric, ExperimentStatus, ExperimentType


@dataclass
class ExperimentResult:
    """Result of an experiment analysis."""
    experiment_id: int
    baseline_metrics: Dict[str, float]
    variant_metrics: Dict[str, float]
    improvement_percentage: float
    is_significant: bool
    p_value: float
    confidence_interval: Tuple[float, float]
    winner: str
    sample_sizes: Dict[str, int]
    recommendation: str


class ExperimentService:
    """Service for managing A/B experiments on query optimizations."""
    
    def __init__(self) -> None:
        self._active_experiments: Dict[int, Experiment] = {}
    
    async def create_experiment(
        self,
        session: AsyncSession,
        name: str,
        description: str,
        experiment_type: ExperimentType,
        baseline_config: Dict[str, Any],
        variant_config: Dict[str, Any],
        target_patterns: Optional[List[int]] = None,
        duration_hours: int = 24,
        baseline_percentage: int = 50
    ) -> Experiment:
        """
        Create a new experiment.
        
        Args:
            session: Database session
            name: Experiment name
            description: Description of what's being tested
            experiment_type: Type of experiment
            baseline_config: Configuration for baseline (no optimization)
            variant_config: Configuration for variant (with optimization)
            target_patterns: Query pattern IDs to include
            duration_hours: How long to run the experiment
            baseline_percentage: Traffic percentage for baseline
            
        Returns:
            Created experiment
        """
        experiment = Experiment(
            name=name,
            description=description,
            experiment_type=experiment_type,
            control_config=baseline_config,
            treatment_config=variant_config,
            target_query_patterns=target_patterns,
            traffic_percentage=baseline_percentage,
            status=ExperimentStatus.DRAFT
        )
        
        session.add(experiment)
        await session.commit()
        await session.refresh(experiment)
        
        return experiment
    
    async def start_experiment(
        self,
        session: AsyncSession,
        experiment_id: int
    ) -> Experiment:
        """Start an experiment."""
        experiment = await session.get(Experiment, experiment_id)
        
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        current_status = cast(ExperimentStatus, experiment.status)
        if current_status != ExperimentStatus.DRAFT:
            raise ValueError(f"Experiment must be in DRAFT status to start")
        
        # Update experiment - SQLAlchemy ORM handles the type conversion at runtime
        experiment.status = ExperimentStatus.RUNNING  # type: ignore[assignment]
        experiment.started_at = datetime.utcnow()  # type: ignore[assignment]
        experiment.ended_at = datetime.utcnow() + timedelta(hours=24)  # type: ignore[assignment]
        
        await session.commit()
        await session.refresh(experiment)
        
        # Add to active experiments cache
        exp_id = cast(int, experiment.id)
        self._active_experiments[exp_id] = experiment
        
        return experiment
    
    async def stop_experiment(
        self,
        session: AsyncSession,
        experiment_id: int,
        status: ExperimentStatus = ExperimentStatus.COMPLETED
    ) -> Experiment:
        """Stop an experiment."""
        experiment = await session.get(Experiment, experiment_id)
        
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment.status = status  # type: ignore[assignment]
        experiment.ended_at = datetime.utcnow()  # type: ignore[assignment]
        
        # Analyze results if completing
        if status == ExperimentStatus.COMPLETED:
            result = await self.analyze_experiment(session, experiment_id)
            experiment.winner = result.winner  # type: ignore[assignment]
            experiment.statistical_significance = result.p_value  # type: ignore[assignment]
        
        await session.commit()
        await session.refresh(experiment)
        
        # Remove from active experiments
        self._active_experiments.pop(experiment_id, None)
        
        return experiment
    
    async def record_metric(
        self,
        session: AsyncSession,
        experiment_id: int,
        variant: str,
        query_hash: str,
        execution_time_ms: float,
        rows_examined: Optional[int] = None,
        rows_returned: Optional[int] = None,
        pattern_id: Optional[int] = None
    ) -> ExperimentMetric:
        """Record a metric for an experiment."""
        metric = ExperimentMetric(
            experiment_id=experiment_id,
            variant=variant,
            query_pattern_id=pattern_id,
            execution_time_ms=execution_time_ms,
            rows_examined=rows_examined,
            rows_returned=rows_returned,
            recorded_at=datetime.utcnow()
        )
        
        session.add(metric)
        await session.commit()
        
        return metric
    
    def assign_variant(
        self,
        experiment: Experiment,
        query_hash: Optional[str] = None
    ) -> str:
        """
        Assign a variant for a query execution.
        
        Uses consistent hashing if query_hash provided,
        otherwise random assignment.
        """
        traffic_pct_raw = experiment.traffic_percentage
        traffic_pct: int = int(traffic_pct_raw) if traffic_pct_raw is not None else 50  # type: ignore[arg-type]
        
        if query_hash:
            # Consistent assignment based on query hash
            hash_value = int(query_hash, 16) % 100
            return "baseline" if hash_value < traffic_pct else "variant"
        else:
            # Random assignment
            return "baseline" if random.randint(1, 100) <= traffic_pct else "variant"
    
    async def get_active_experiment(
        self,
        session: AsyncSession,
        pattern_id: Optional[int] = None
    ) -> Optional[Experiment]:
        """Get an active experiment for a query pattern."""
        # Check in-memory cache first
        for exp in self._active_experiments.values():
            exp_status = cast(ExperimentStatus, exp.status)
            if exp_status == ExperimentStatus.RUNNING:
                target_patterns = cast(Optional[List[int]], exp.target_query_patterns)
                if pattern_id is None or (
                    target_patterns is not None and 
                    pattern_id in target_patterns
                ):
                    return exp
        
        # Query database
        query = select(Experiment).where(
            Experiment.status == ExperimentStatus.RUNNING
        )
        
        result = await session.execute(query)
        experiments = result.scalars().all()
        
        for exp in experiments:
            target_patterns = cast(Optional[List[int]], exp.target_query_patterns)
            if pattern_id is None or (
                target_patterns is not None and 
                pattern_id in target_patterns
            ):
                exp_id = cast(int, exp.id)
                self._active_experiments[exp_id] = exp
                return exp
        
        return None
    
    async def analyze_experiment(
        self,
        session: AsyncSession,
        experiment_id: int
    ) -> ExperimentResult:
        """
        Analyze experiment results and determine winner.
        
        Uses statistical testing to determine significance.
        """
        # Get all metrics for the experiment
        query = select(ExperimentMetric).where(
            ExperimentMetric.experiment_id == experiment_id
        )
        result = await session.execute(query)
        metrics = result.scalars().all()
        
        # Separate by variant - extract actual values from ORM objects
        baseline_times: List[float] = []
        variant_times: List[float] = []
        
        for m in metrics:
            variant_val = str(m.variant) if m.variant is not None else ""
            exec_time_raw = m.execution_time_ms
            exec_time = float(exec_time_raw) if exec_time_raw is not None else 0.0  # type: ignore[arg-type]
            if variant_val == "baseline":
                baseline_times.append(exec_time)
            elif variant_val == "variant":
                variant_times.append(exec_time)
        
        if not baseline_times or not variant_times:
            return ExperimentResult(
                experiment_id=experiment_id,
                baseline_metrics={},
                variant_metrics={},
                improvement_percentage=0,
                is_significant=False,
                p_value=1.0,
                confidence_interval=(0, 0),
                winner="insufficient_data",
                sample_sizes={"baseline": len(baseline_times), "variant": len(variant_times)},
                recommendation="Insufficient data to determine winner"
            )
        
        # Calculate statistics
        baseline_stats = self._calculate_stats(baseline_times)
        variant_stats = self._calculate_stats(variant_times)
        
        # Perform t-test - returns (statistic, pvalue) tuple
        ttest_result = stats.ttest_ind(baseline_times, variant_times)
        p_value = float(ttest_result[1])  # type: ignore[arg-type]
        
        # Calculate improvement
        improvement = ((baseline_stats["mean"] - variant_stats["mean"]) / baseline_stats["mean"]) * 100
        
        # Calculate confidence interval for the difference
        diff_mean = baseline_stats["mean"] - variant_stats["mean"]
        pooled_se = np.sqrt(
            baseline_stats["std"]**2 / len(baseline_times) + 
            variant_stats["std"]**2 / len(variant_times)
        )
        ci_low = diff_mean - 1.96 * pooled_se
        ci_high = diff_mean + 1.96 * pooled_se
        
        # Determine winner
        is_significant = p_value < 0.05
        if not is_significant:
            winner = "no_significant_difference"
            recommendation = "No statistically significant difference detected. Consider running longer."
        elif variant_stats["mean"] < baseline_stats["mean"]:
            winner = "variant"
            recommendation = f"Variant is {improvement:.1f}% faster. Consider applying optimization."
        else:
            winner = "baseline"
            recommendation = f"Optimization shows {-improvement:.1f}% regression. Do not apply."
        
        return ExperimentResult(
            experiment_id=experiment_id,
            baseline_metrics=baseline_stats,
            variant_metrics=variant_stats,
            improvement_percentage=improvement,
            is_significant=is_significant,
            p_value=p_value,
            confidence_interval=(ci_low, ci_high),
            winner=winner,
            sample_sizes={"baseline": len(baseline_times), "variant": len(variant_times)},
            recommendation=recommendation
        )
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values."""
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "count": len(values)
        }
    
    async def get_experiment_progress(
        self,
        session: AsyncSession,
        experiment_id: int
    ) -> Dict[str, Any]:
        """Get progress of a running experiment."""
        experiment = await session.get(Experiment, experiment_id)
        
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Count metrics
        query = select(
            ExperimentMetric.variant,
            func.count(ExperimentMetric.id),
            func.avg(ExperimentMetric.execution_time_ms)
        ).where(
            ExperimentMetric.experiment_id == experiment_id
        ).group_by(ExperimentMetric.variant)
        
        result = await session.execute(query)
        rows = result.fetchall()
        
        metrics_by_variant = {row[0]: {"count": row[1], "avg_time": row[2]} for row in rows}
        
        # Calculate time progress - cast ORM datetime columns
        started = cast(Optional[datetime], experiment.started_at)
        ended = cast(Optional[datetime], experiment.ended_at)
        
        if started and ended:
            now = datetime.utcnow()
            total_duration = (ended - started).total_seconds()
            elapsed = (now - started).total_seconds()
            time_progress = min(100.0, (elapsed / total_duration) * 100)
        else:
            time_progress = 0.0
        
        exp_status = cast(Optional[ExperimentStatus], experiment.status)
        
        return {
            "experiment_id": experiment_id,
            "status": exp_status.value if exp_status else None,
            "time_progress_percentage": time_progress,
            "metrics_collected": metrics_by_variant,
            "start_time": started,
            "end_time": ended,
            "estimated_completion": ended
        }
    
    async def list_experiments(
        self,
        session: AsyncSession,
        status: Optional[ExperimentStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Experiment]:
        """List experiments with optional filtering."""
        query = select(Experiment)
        
        if status:
            query = query.where(Experiment.status == status)
        
        query = query.order_by(Experiment.created_at.desc())
        query = query.limit(limit).offset(offset)
        
        result = await session.execute(query)
        return list(result.scalars().all())
    
    async def check_experiment_completion(self, session: AsyncSession):
        """Check and complete any experiments that have reached their end time."""
        now = datetime.utcnow()
        
        query = select(Experiment).where(
            Experiment.status == ExperimentStatus.RUNNING,
            Experiment.ended_at <= now
        )
        
        result = await session.execute(query)
        experiments = result.scalars().all()
        
        for experiment in experiments:
            exp_id = cast(Optional[int], experiment.id)
            if exp_id is not None:
                await self.stop_experiment(session, exp_id, ExperimentStatus.COMPLETED)
    
    def analyze_experiment_results(
        self,
        experiment: Experiment,
        metrics: List[ExperimentMetric]
    ) -> Dict[str, Any]:
        """
        Analyze experiment results for API endpoint.
        
        Returns structured analysis with control/treatment metrics.
        """
        # Separate by variant
        control_times: List[float] = []
        treatment_times: List[float] = []
        
        for m in metrics:
            variant_raw = m.variant
            variant_val = str(variant_raw) if variant_raw is not None else ""
            exec_time_raw = m.execution_time_ms
            exec_time = float(exec_time_raw) if exec_time_raw is not None else 0.0  # type: ignore[arg-type]
            if variant_val == "control":
                control_times.append(exec_time)
            elif variant_val == "treatment":
                treatment_times.append(exec_time)
        
        # Calculate metrics
        control_metrics = self._calculate_stats(control_times) if control_times else {}
        treatment_metrics = self._calculate_stats(treatment_times) if treatment_times else {}
        
        # Statistical analysis
        statistical_analysis: Dict[str, Any] = {}
        recommendation = "Insufficient data"
        confidence_level = 0.0
        
        if control_times and treatment_times:
            ttest_result = stats.ttest_ind(control_times, treatment_times)
            t_stat = float(ttest_result[0])  # type: ignore[arg-type]
            p_value = float(ttest_result[1])  # type: ignore[arg-type]
            is_significant = p_value < 0.05
            
            # Calculate improvement
            if control_metrics.get("mean", 0) > 0:
                improvement = ((control_metrics["mean"] - treatment_metrics.get("mean", 0)) / 
                             control_metrics["mean"]) * 100
            else:
                improvement = 0
            
            statistical_analysis = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "is_significant": is_significant,
                "improvement_percentage": improvement
            }
            
            confidence_level = 1 - p_value
            
            if not is_significant:
                recommendation = "No statistically significant difference. Consider running longer."
            elif treatment_metrics.get("mean", float('inf')) < control_metrics.get("mean", 0):
                recommendation = f"Treatment is {improvement:.1f}% faster. Recommend applying optimization."
            else:
                recommendation = f"Control performs better. Do not apply optimization."
        
        return {
            "control_metrics": control_metrics,
            "treatment_metrics": treatment_metrics,
            "statistical_analysis": statistical_analysis,
            "recommendation": recommendation,
            "confidence_level": confidence_level,
            "sample_sizes": {
                "control": len(control_times),
                "treatment": len(treatment_times)
            }
        }
