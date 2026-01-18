"""
Experiment-related API endpoints for A/B testing.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from typing import List, Optional, cast
from datetime import datetime, timedelta

from app.core.database import get_db
from app.models.experiment import Experiment, ExperimentMetric, ExperimentStatus
from app.schemas.experiment import (
    ExperimentCreate, ExperimentResponse, ExperimentList,
    ExperimentMetricCreate, ExperimentMetricResponse,
    ExperimentResultResponse
)
from app.services.experiment_service import ExperimentService

router = APIRouter(prefix="/experiments", tags=["Experiments"])

# Initialize service
experiment_service = ExperimentService()


@router.get("/", response_model=ExperimentList)
async def list_experiments(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: Optional[ExperimentStatus] = Query(None, alias="status"),
    db: AsyncSession = Depends(get_db)
):
    """
    List experiments with filtering and pagination.
    
    - **status**: Filter by experiment status
    """
    query = select(Experiment)
    
    if status_filter:
        query = query.where(Experiment.status == status_filter)
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination and ordering
    query = query.order_by(desc(Experiment.created_at))
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    items = result.scalars().all()
    
    return ExperimentList(
        items=[ExperimentResponse.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if total > 0 else 1
    )


@router.get("/active")
async def list_active_experiments(
    db: AsyncSession = Depends(get_db)
):
    """List currently running experiments."""
    query = select(Experiment).where(
        Experiment.status == ExperimentStatus.RUNNING
    ).order_by(desc(Experiment.started_at))
    
    result = await db.execute(query)
    experiments = result.scalars().all()
    
    return {
        "active_experiments": [
            ExperimentResponse.model_validate(exp) for exp in experiments
        ],
        "count": len(experiments)
    }


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific experiment by ID."""
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found"
        )
    return ExperimentResponse.model_validate(experiment)


@router.post("/", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    experiment_data: ExperimentCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new A/B test experiment.
    
    This sets up an experiment to compare query optimization strategies.
    """
    experiment = Experiment(
        name=experiment_data.name,
        description=experiment_data.description,
        experiment_type=experiment_data.experiment_type,
        control_config=experiment_data.control_config,
        treatment_config=experiment_data.treatment_config,
        target_query_patterns=experiment_data.target_query_patterns,
        traffic_percentage=experiment_data.traffic_percentage,
        required_sample_size=experiment_data.required_sample_size,
        status=ExperimentStatus.DRAFT
    )
    
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)
    
    return ExperimentResponse.model_validate(experiment)


@router.post("/{experiment_id}/start")
async def start_experiment(
    experiment_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Start an experiment."""
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found"
        )
    
    exp_status = cast(str, experiment.status)
    if exp_status == ExperimentStatus.RUNNING.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Experiment is already running"
        )
    
    if exp_status in [ExperimentStatus.COMPLETED.value, ExperimentStatus.ABORTED.value]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot start experiment in {exp_status} status"
        )
    
    experiment.status = ExperimentStatus.RUNNING.value  # type: ignore[assignment]
    experiment.started_at = datetime.utcnow()  # type: ignore[assignment]
    
    await db.commit()
    
    return {
        "success": True,
        "experiment_id": experiment_id,
        "status": "running",
        "started_at": experiment.started_at
    }


@router.post("/{experiment_id}/stop")
async def stop_experiment(
    experiment_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Stop a running experiment."""
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found"
        )
    
    exp_status = cast(str, experiment.status)
    if exp_status != ExperimentStatus.RUNNING.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Experiment is not running"
        )
    
    experiment.status = ExperimentStatus.COMPLETED.value  # type: ignore[assignment]
    experiment.ended_at = datetime.utcnow()  # type: ignore[assignment]
    
    await db.commit()
    
    return {
        "success": True,
        "experiment_id": experiment_id,
        "status": "completed",
        "ended_at": experiment.ended_at
    }


@router.post("/{experiment_id}/abort")
async def abort_experiment(
    experiment_id: int,
    reason: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Abort an experiment."""
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found"
        )
    
    exp_status = cast(str, experiment.status)
    if exp_status in [ExperimentStatus.COMPLETED.value, ExperimentStatus.ABORTED.value]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot abort experiment in {exp_status} status"
        )
    
    experiment.status = ExperimentStatus.ABORTED.value  # type: ignore[assignment]
    experiment.ended_at = datetime.utcnow()  # type: ignore[assignment]
    
    await db.commit()
    
    return {
        "success": True,
        "experiment_id": experiment_id,
        "status": "aborted",
        "reason": reason,
        "ended_at": experiment.ended_at
    }


@router.post("/{experiment_id}/metrics", response_model=ExperimentMetricResponse)
async def record_experiment_metric(
    experiment_id: int,
    metric_data: ExperimentMetricCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Record a metric for an experiment.
    
    This is called to log performance data during the experiment.
    """
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found"
        )
    
    exp_status = cast(str, experiment.status)
    if exp_status != ExperimentStatus.RUNNING.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only record metrics for running experiments"
        )
    
    metric = ExperimentMetric(
        experiment_id=experiment_id,
        variant=metric_data.variant,
        query_pattern_id=metric_data.query_pattern_id,
        query_log_id=metric_data.query_log_id,
        execution_time_ms=metric_data.execution_time_ms,
        rows_examined=metric_data.rows_examined,
        rows_returned=metric_data.rows_returned,
        cache_hit=metric_data.cache_hit,
        error_occurred=metric_data.error_occurred,
        additional_metrics=metric_data.additional_metrics,
        recorded_at=datetime.utcnow()
    )
    
    db.add(metric)
    await db.commit()
    await db.refresh(metric)
    
    return ExperimentMetricResponse.model_validate(metric)


@router.get("/{experiment_id}/metrics")
async def get_experiment_metrics(
    experiment_id: int,
    variant: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get metrics for an experiment."""
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found"
        )
    
    query = select(ExperimentMetric).where(
        ExperimentMetric.experiment_id == experiment_id
    )
    
    if variant:
        query = query.where(ExperimentMetric.variant == variant)
    
    result = await db.execute(query)
    metrics = result.scalars().all()
    
    return {
        "experiment_id": experiment_id,
        "metrics": [ExperimentMetricResponse.model_validate(m) for m in metrics],
        "count": len(metrics)
    }


@router.get("/{experiment_id}/results", response_model=ExperimentResultResponse)
async def get_experiment_results(
    experiment_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get analyzed results for an experiment.
    
    This performs statistical analysis on the collected metrics.
    """
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found"
        )
    
    # Get all metrics
    query = select(ExperimentMetric).where(
        ExperimentMetric.experiment_id == experiment_id
    )
    result = await db.execute(query)
    metrics = list(result.scalars().all())
    
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No metrics recorded for this experiment"
        )
    
    # Analyze the results
    analysis = experiment_service.analyze_experiment_results(experiment, metrics)
    
    # Cast SQLAlchemy columns to Python types (cast even if None at runtime)
    exp_name = cast(str, experiment.name)
    exp_status = cast(str, experiment.status)
    started_at = cast(datetime, experiment.started_at)
    ended_at = cast(datetime, experiment.ended_at)
    
    return ExperimentResultResponse(
        experiment_id=experiment_id,
        experiment_name=exp_name,
        status=exp_status,
        started_at=started_at,  # type: ignore[arg-type]
        ended_at=ended_at,  # type: ignore[arg-type]
        control_metrics=analysis["control_metrics"],
        treatment_metrics=analysis["treatment_metrics"],
        statistical_analysis=analysis["statistical_analysis"],
        recommendation=analysis["recommendation"],
        confidence_level=analysis["confidence_level"],
        sample_sizes=analysis["sample_sizes"]
    )


@router.get("/{experiment_id}/summary")
async def get_experiment_summary(
    experiment_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a quick summary of experiment progress."""
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found"
        )
    
    # Count metrics per variant
    control_count_query = select(func.count()).select_from(ExperimentMetric).where(
        ExperimentMetric.experiment_id == experiment_id,
        ExperimentMetric.variant == "control"
    )
    control_result = await db.execute(control_count_query)
    control_count = control_result.scalar() or 0
    
    treatment_count_query = select(func.count()).select_from(ExperimentMetric).where(
        ExperimentMetric.experiment_id == experiment_id,
        ExperimentMetric.variant == "treatment"
    )
    treatment_result = await db.execute(treatment_count_query)
    treatment_count = treatment_result.scalar() or 0
    
    # Average execution times
    control_avg_query = select(func.avg(ExperimentMetric.execution_time_ms)).where(
        ExperimentMetric.experiment_id == experiment_id,
        ExperimentMetric.variant == "control"
    )
    control_avg_result = await db.execute(control_avg_query)
    control_avg = control_avg_result.scalar() or 0
    
    treatment_avg_query = select(func.avg(ExperimentMetric.execution_time_ms)).where(
        ExperimentMetric.experiment_id == experiment_id,
        ExperimentMetric.variant == "treatment"
    )
    treatment_avg_result = await db.execute(treatment_avg_query)
    treatment_avg = treatment_avg_result.scalar() or 0
    
    # Calculate progress
    total_samples = control_count + treatment_count
    required_sample_size = cast(int, experiment.required_sample_size)
    progress = (total_samples / required_sample_size * 100) if required_sample_size else 0
    
    # Calculate improvement
    improvement = ((control_avg - treatment_avg) / control_avg * 100) if control_avg > 0 else 0
    
    # Get status and timestamps - cast to Python types
    exp_status = cast(str, experiment.status)
    exp_name = cast(str, experiment.name)
    started_at = cast(datetime, experiment.started_at)
    ended_at = cast(datetime, experiment.ended_at)
    
    # Calculate running hours
    running_hours = 0.0
    if started_at:
        end_time = ended_at or datetime.utcnow()
        running_hours = round((end_time - started_at).total_seconds() / 3600, 2)
    
    return {
        "experiment_id": experiment_id,
        "name": exp_name,
        "status": exp_status,
        "progress_percentage": min(100, round(float(progress), 2)),
        "sample_sizes": {
            "control": control_count,
            "treatment": treatment_count,
            "total": total_samples,
            "required": required_sample_size
        },
        "preliminary_results": {
            "control_avg_ms": round(float(control_avg), 2),
            "treatment_avg_ms": round(float(treatment_avg), 2),
            "improvement_percentage": round(float(improvement), 2)
        },
        "duration": {
            "started_at": started_at,
            "ended_at": ended_at,
            "running_hours": running_hours
        }
    }


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_experiment(
    experiment_id: int,
    force: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete an experiment.
    
    Use force=True to delete a running experiment.
    """
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found"
        )
    
    exp_status = cast(str, experiment.status)
    if exp_status == ExperimentStatus.RUNNING.value and not force:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete running experiment. Use force=True to override."
        )
    
    # Delete associated metrics
    delete_metrics_query = select(ExperimentMetric).where(
        ExperimentMetric.experiment_id == experiment_id
    )
    metrics_result = await db.execute(delete_metrics_query)
    for metric in metrics_result.scalars().all():
        await db.delete(metric)
    
    await db.delete(experiment)
    await db.commit()
