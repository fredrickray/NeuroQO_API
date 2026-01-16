"""
Model Manager - Manages ML model lifecycle, training, versioning, and monitoring.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

import numpy as np
from sqlalchemy.future import select
from fastapi.concurrency import run_in_threadpool
from app.models.query import QueryLog, OptimizationResult
from app.ml.feature_extractor import QueryFeatureExtractor
from app.ml.query_classifier import QueryClassifier, OptimizationPriority
from app.ml.performance_predictor import PerformancePredictor
from app.ml.optimization_recommender import OptimizationRecommender, OptimizationType
from app.core.config import settings


# =========================
# Dataclasses
# =========================

@dataclass
class ModelInfo:
    model_id: str
    model_type: str
    version: str
    trained_at: datetime
    training_samples: int
    metrics: Dict[str, float]
    feature_count: int
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["trained_at"] = self.trained_at.isoformat()
        return data


@dataclass
class DriftMetrics:
    feature_drift_score: float
    prediction_drift_score: float
    is_drift_detected: bool
    features_with_drift: List[str]
    recommendation: str


@dataclass
class ModelMonitoringStats:
    model_id: str
    total_predictions: int
    avg_confidence: float
    low_confidence_count: int
    high_confidence_count: int
    prediction_distribution: Dict[str, int]
    last_prediction_at: Optional[datetime]
    drift_metrics: Optional[DriftMetrics]


# =========================
# Model Manager
# =========================

class ModelManager:
    MODEL_TYPES = {"classifier", "predictor", "recommender"}

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir or settings.MODEL_PATH)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.classifier: Optional[QueryClassifier] = None
        self.performance_predictor: Optional[PerformancePredictor] = None
        self.recommender: Optional[OptimizationRecommender] = None

        self.feature_extractor = QueryFeatureExtractor()

        self.model_registry: Dict[str, Dict[str, ModelInfo]] = {}
        self._load_registry()

        self._prediction_log: List[Dict[str, Any]] = []

        self._baseline_feature_stats: Optional[Dict[str, Tuple[float, float]]] = None

    # =========================
    # Registry Persistence
    # =========================

    def _load_registry(self) -> None:
        path = self.model_dir / "registry.json"
        if not path.exists():
            return

        with open(path, "r") as f:
            raw = json.load(f)

        for model_type, versions in raw.items():
            self.model_registry[model_type] = {}
            for model_id, info in versions.items():
                info["trained_at"] = datetime.fromisoformat(info["trained_at"])
                self.model_registry[model_type][model_id] = ModelInfo(**info)

    def _save_registry(self) -> None:
        data = {
            mt: {mid: info.to_dict() for mid, info in versions.items()}
            for mt, versions in self.model_registry.items()
        }
        with open(self.model_dir / "registry.json", "w") as f:
            json.dump(data, f, indent=2)

    def _register_model(self, info: ModelInfo) -> None:
        self.model_registry.setdefault(info.model_type, {})
        # deactivate all previous versions
        for model in self.model_registry[info.model_type].values():
            model.is_active = False
        self.model_registry[info.model_type][info.model_id] = info
        self._save_registry()

    def _get_latest_model(self, model_type: str) -> Optional[ModelInfo]:
        versions = self.model_registry.get(model_type, {})
        active = [m for m in versions.values() if m.is_active]
        return active[0] if active else None

    # =========================
    # Utilities
    # =========================

    @staticmethod
    def hash_query(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

    def log_prediction(self, model: str, query: str, confidence: float, result: Any) -> None:
        self._prediction_log.append({
            "model": model,
            "query_hash": self.hash_query(query),
            "confidence": confidence,
            "result": result,
            "timestamp": datetime.utcnow(),
        })
        self._prediction_log = self._prediction_log[-10_000:]  # keep last 10k

    # =========================
    # DB Fetch
    # =========================

    async def fetch_classifier_training_data(self, db):
        result = await db.execute(select(QueryLog))
        rows = result.scalars().all()
        return (
            [r.query_text for r in rows],
            [r.complexity for r in rows],
            [OptimizationPriority(r.priority) if hasattr(r, "priority") else OptimizationPriority.MEDIUM for r in rows],
        )

    async def fetch_predictor_training_data(self, db):
        result = await db.execute(select(QueryLog))
        rows = result.scalars().all()
        return [r.query_text for r in rows], [r.execution_time_ms for r in rows]

    async def fetch_recommender_training_data(self, db):
        result = await db.execute(select(OptimizationResult))
        rows = result.scalars().all()
        queries = [r.original_query for r in rows]
        applied = [r.optimization_rules_applied if r.optimization_rules_applied else [] for r in rows]
        return queries, applied

    # =========================
    # Training
    # =========================

    async def train_model(self, model_type: str, db) -> Dict[str, Any]:
        if model_type not in self.MODEL_TYPES:
            return {"model_type": model_type, "status": "error", "message": "Invalid model type"}

        if model_type == "classifier":
            queries, categories, priorities = await self.fetch_classifier_training_data(db)
            metrics = await run_in_threadpool(self._train_classifier_sync, queries, categories, priorities)
            return {"model_type": model_type, "status": "trained", "metrics": metrics}

        if model_type == "predictor":
            queries, execution_times = await self.fetch_predictor_training_data(db)
            metrics = await run_in_threadpool(self._train_predictor_sync, queries, execution_times)
            return {"model_type": model_type, "status": "trained", "metrics": metrics}

        if model_type == "recommender":
            queries, applied = await self.fetch_recommender_training_data(db)
            metrics = await run_in_threadpool(self._train_recommender_sync, queries, applied)
            return {"model_type": model_type, "status": "trained", "metrics": metrics}

        return {"model_type": model_type, "status": "error", "message": "Unhandled model type"}

    def _train_classifier_sync(self, queries: List[str], categories: List[Any], priorities: List[Any]) -> Dict[str, float]:
        self.classifier = QueryClassifier()
        metrics = self.classifier.train(queries, categories, priorities)
        self.classifier.save(str(self.model_dir / "classifier.pkl"))

        info = ModelInfo(
            model_id=f"classifier_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            model_type="classifier",
            version=self.classifier.model_version,
            trained_at=datetime.utcnow(),
            training_samples=len(queries),
            metrics=metrics,
            feature_count=len(self.feature_extractor.get_feature_names()),
        )
        self._register_model(info)
        self._set_baseline_stats(queries)
        return metrics

    def _train_predictor_sync(self, queries: List[str], execution_times: List[float]) -> Dict[str, float]:
        self.performance_predictor = PerformancePredictor()
        metrics_obj = self.performance_predictor.train(queries, execution_times)
        # Only include numeric values in metrics
        metrics: Dict[str, float] = {k: float(v) for k, v in vars(metrics_obj).items() if isinstance(v, (int, float))}
        self.performance_predictor.save(str(self.model_dir / "predictor.pkl"))

        info = ModelInfo(
            model_id=f"predictor_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            model_type="predictor",
            version=self.performance_predictor.model_version,
            trained_at=datetime.utcnow(),
            training_samples=len(queries),
            metrics=metrics,
            feature_count=len(self.feature_extractor.get_feature_names()),
        )
        self._register_model(info)
        self._set_baseline_stats(queries)
        return metrics

    def _train_recommender_sync(self, queries: List[str], applied_optimizations: List[List[str]]) -> Dict[str, float]:
        
        self.recommender = OptimizationRecommender()
        
        # Convert string optimization types to enum, skipping invalid values
        valid_opt_values = {e.value for e in OptimizationType}
        applied_optimizations_typed = []
        for opts in applied_optimizations:
            valid_opts = []
            for opt in opts:
                if opt in valid_opt_values:
                    valid_opts.append(OptimizationType(opt))
                else:
                    print(f"Warning: Skipping unknown optimization type '{opt}'")
            applied_optimizations_typed.append(valid_opts)
        metrics = self.recommender.train(queries, applied_optimizations_typed)
        self.recommender.save(str(self.model_dir / "recommender.pkl"))

        info = ModelInfo(
            model_id=f"recommender_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            model_type="recommender",
            version=self.recommender.model_version,
            trained_at=datetime.utcnow(),
            training_samples=len(queries),
            metrics=metrics,
            feature_count=len(self.feature_extractor.get_feature_names()),
        )
        self._register_model(info)
        self._set_baseline_stats(queries)
        return metrics

    # =========================
    # Drift Detection
    # =========================

    def _set_baseline_stats(self, queries: List[str]) -> None:
        if not queries:
            return
        X, _ = self.feature_extractor.extract_batch(queries)
        self._baseline_feature_stats = {name: (float(np.mean(X[:, i])), float(np.std(X[:, i])))
                                        for i, name in enumerate(self.feature_extractor.get_feature_names())}

    def detect_drift(self, recent_queries: List[str]) -> DriftMetrics:
        if not self._baseline_feature_stats or not recent_queries:
            return DriftMetrics(0.0, 0.0, False, [], "No baseline or recent queries")

        X, _ = self.feature_extractor.extract_batch(recent_queries)
        scores: List[float] = []
        drifted: List[str] = []

        for i, name in enumerate(self.feature_extractor.get_feature_names()):
            base_mean, base_std = self._baseline_feature_stats[name]
            current_mean = float(np.mean(X[:, i]))
            z = abs(current_mean - base_mean) / base_std if base_std else 0.0
            scores.append(z)
            if z > 2:
                drifted.append(name)

        feature_score = float(np.mean(scores))
        detected = feature_score > 1.5 or len(drifted) > 5

        return DriftMetrics(
            feature_drift_score=feature_score,
            prediction_drift_score=0.0,
            is_drift_detected=detected,
            features_with_drift=drifted,
            recommendation="Retrain recommended" if detected else "Model stable"
        )

    # =========================
    # Monitoring & Metrics
    # =========================

    def get_model_metrics(self, model_type: str) -> Dict[str, float]:
        model = self._get_latest_model(model_type)
        return model.metrics if model else {}

    def _prediction_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for p in self._prediction_log:
            key = str(p["result"])
            dist[key] = dist.get(key, 0) + 1
        return dist

    def get_model_monitoring_stats(self, model_type: str) -> Optional[ModelMonitoringStats]:
        model = self._get_latest_model(model_type)
        if not model:
            return None

        predictions = [p for p in self._prediction_log if p["model"] == model_type]
        if not predictions:
            return ModelMonitoringStats(
                model_id=model.model_id,
                total_predictions=0,
                avg_confidence=0.0,
                low_confidence_count=0,
                high_confidence_count=0,
                prediction_distribution={},
                last_prediction_at=None,
                drift_metrics=None,
            )

        confidences = [p["confidence"] for p in predictions]

        drift = self.detect_drift([p["query_hash"] for p in predictions[-50:]]) if len(predictions) >= 50 else None

        return ModelMonitoringStats(
            model_id=model.model_id,
            total_predictions=len(predictions),
            avg_confidence=float(np.mean(confidences)),
            low_confidence_count=sum(1 for c in confidences if c < 0.5),
            high_confidence_count=sum(1 for c in confidences if c > 0.8),
            prediction_distribution=self._prediction_distribution(),
            last_prediction_at=predictions[-1]["timestamp"],
            drift_metrics=drift,
        )

    def get_monitoring_summary(self) -> Dict[str, Any]:
        return {
            "total_predictions": len(self._prediction_log),
            "models_active": sum(1 for models in self.model_registry.values() for m in models.values() if m.is_active),
            "models_total": sum(len(models) for models in self.model_registry.values()),
        }

    def get_model_info(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        if model_type:
            info = self._get_latest_model(model_type)
            return info.to_dict() if info else {}
        return {mt: {mid: info.to_dict() for mid, info in versions.items()} for mt, versions in self.model_registry.items()}

    # =========================
    # Version Utilities
    # =========================

    def list_model_versions(self, model_type: str) -> List[str]:
        return list(self.model_registry.get(model_type, {}).keys())

    def get_current_version(self, model_type: str) -> Optional[str]:
        latest = self._get_latest_model(model_type)
        return latest.version if latest else None
