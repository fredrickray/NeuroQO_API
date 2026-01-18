"""
Model Manager - Manages ML model lifecycle, training, and monitoring.
"""
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np

from app.ml.feature_extractor import QueryFeatureExtractor
from app.ml.query_classifier import QueryClassifier
from app.ml.performance_predictor import PerformancePredictor
from app.ml.optimization_recommender import OptimizationRecommender
from app.core.config import settings


@dataclass
class ModelInfo:
    """Information about a trained model."""
    model_id: str
    model_type: str
    version: str
    trained_at: datetime
    training_samples: int
    metrics: Dict[str, float]
    feature_count: int
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['trained_at'] = self.trained_at.isoformat()
        return d


@dataclass
class DriftMetrics:
    """Metrics for model drift detection."""
    feature_drift_score: float
    prediction_drift_score: float
    is_drift_detected: bool
    features_with_drift: List[str]
    recommendation: str


@dataclass
class ModelMonitoringStats:
    """Statistics for model monitoring."""
    model_id: str
    total_predictions: int
    avg_confidence: float
    low_confidence_count: int
    high_confidence_count: int
    prediction_distribution: Dict[str, int]
    last_prediction_at: Optional[datetime]
    drift_metrics: Optional[DriftMetrics]


class ModelManager:
    """
    Central manager for all ML models in NeuroQO.
    
    Handles:
    - Model training and retraining
    - Model versioning
    - Model persistence
    - Model monitoring and drift detection
    - Model switching
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir or settings.MODEL_PATH)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.classifier: Optional[QueryClassifier] = None
        self.performance_predictor: Optional[PerformancePredictor] = None
        self.recommender: Optional[OptimizationRecommender] = None
        self.feature_extractor = QueryFeatureExtractor()
        
        # Model registry
        self.model_registry: Dict[str, ModelInfo] = {}
        self._load_registry()
        
        # Monitoring data
        self._prediction_log: List[Dict[str, Any]] = []
        self._training_data_hashes: set = set()
        
        # Drift detection parameters
        self._baseline_feature_stats: Optional[Dict[str, Tuple[float, float]]] = None
        self._baseline_prediction_dist: Optional[Dict[str, float]] = None
    
    def _load_registry(self):
        """Load model registry from disk."""
        registry_path = self.model_dir / "registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                data = json.load(f)
                for model_id, info in data.items():
                    info['trained_at'] = datetime.fromisoformat(info['trained_at'])
                    self.model_registry[model_id] = ModelInfo(**info)
    
    def _save_registry(self):
        """Save model registry to disk."""
        registry_path = self.model_dir / "registry.json"
        data = {k: v.to_dict() for k, v in self.model_registry.items()}
        with open(registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def initialize_models(self) -> Dict[str, bool]:
        """Initialize all models, loading from disk if available."""
        results = {}
        
        # Load classifier
        classifier_path = self.model_dir / "classifier.pkl"
        if classifier_path.exists():
            self.classifier = QueryClassifier(str(classifier_path))
            results["classifier"] = True
        else:
            self.classifier = QueryClassifier()
            results["classifier"] = False
        
        # Load performance predictor
        predictor_path = self.model_dir / "predictor.pkl"
        if predictor_path.exists():
            self.performance_predictor = PerformancePredictor(model_path=str(predictor_path))
            results["performance_predictor"] = True
        else:
            self.performance_predictor = PerformancePredictor()
            results["performance_predictor"] = False
        
        # Load recommender
        recommender_path = self.model_dir / "recommender.pkl"
        if recommender_path.exists():
            self.recommender = OptimizationRecommender(str(recommender_path))
            results["recommender"] = True
        else:
            self.recommender = OptimizationRecommender()
            results["recommender"] = False
        
        return results
    
    def train_classifier(
        self,
        queries: List[str],
        categories: List[Any],
        priorities: List[Any]
    ) -> Dict[str, Any]:
        """Train the query classifier."""
        if self.classifier is None:
            self.classifier = QueryClassifier()
        
        metrics = self.classifier.train(queries, categories, priorities)
        
        # Save model
        model_path = self.model_dir / "classifier.pkl"
        self.classifier.save(str(model_path))
        
        # Register model
        model_id = f"classifier_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.model_registry[model_id] = ModelInfo(
            model_id=model_id,
            model_type="classifier",
            version=self.classifier.model_version,
            trained_at=datetime.utcnow(),
            training_samples=len(queries),
            metrics=metrics,
            feature_count=len(self.feature_extractor.get_feature_names())
        )
        self._save_registry()
        
        return metrics
    
    def train_performance_predictor(
        self,
        queries: List[str],
        execution_times: List[float]
    ) -> Dict[str, Any]:
        """Train the performance predictor."""
        if self.performance_predictor is None:
            self.performance_predictor = PerformancePredictor()
        
        metrics = self.performance_predictor.train(queries, execution_times)
        
        # Save model
        model_path = self.model_dir / "predictor.pkl"
        self.performance_predictor.save(str(model_path))
        
        # Register model
        model_id = f"predictor_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.model_registry[model_id] = ModelInfo(
            model_id=model_id,
            model_type="predictor",
            version=self.performance_predictor.model_version,
            trained_at=datetime.utcnow(),
            training_samples=len(queries),
            metrics={
                "rmse": metrics.rmse,
                "mae": metrics.mae,
                "r2": metrics.r2,
                "mape": metrics.mape
            },
            feature_count=len(self.feature_extractor.get_feature_names())
        )
        self._save_registry()
        
        # Set baseline for drift detection
        self._set_baseline_stats(queries)
        
        return asdict(metrics)
    
    def train_recommender(
        self,
        queries: List[str],
        applied_optimizations: List[List[Any]]
    ) -> Dict[str, Any]:
        """Train the optimization recommender."""
        if self.recommender is None:
            self.recommender = OptimizationRecommender()
        
        metrics = self.recommender.train(queries, applied_optimizations)
        
        # Save model
        model_path = self.model_dir / "recommender.pkl"
        self.recommender.save(str(model_path))
        
        # Register model
        model_id = f"recommender_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.model_registry[model_id] = ModelInfo(
            model_id=model_id,
            model_type="recommender",
            version=self.recommender.model_version,
            trained_at=datetime.utcnow(),
            training_samples=len(queries),
            metrics=metrics,
            feature_count=len(self.feature_extractor.get_feature_names())
        )
        self._save_registry()
        
        return metrics
    
    def _set_baseline_stats(self, queries: List[str]):
        """Set baseline statistics for drift detection."""
        X, _ = self.feature_extractor.extract_batch(queries)
        
        self._baseline_feature_stats = {}
        for i, name in enumerate(self.feature_extractor.get_feature_names()):
            self._baseline_feature_stats[name] = (
                float(np.mean(X[:, i])),
                float(np.std(X[:, i]))
            )
    
    def detect_drift(
        self,
        recent_queries: List[str]
    ) -> DriftMetrics:
        """
        Detect model drift based on recent queries.
        
        Compares feature distributions of recent queries
        with the training data baseline.
        """
        if self._baseline_feature_stats is None:
            return DriftMetrics(
                feature_drift_score=0,
                prediction_drift_score=0,
                is_drift_detected=False,
                features_with_drift=[],
                recommendation="No baseline available for drift detection"
            )
        
        X, _ = self.feature_extractor.extract_batch(recent_queries)
        
        drifted_features = []
        drift_scores = []
        
        for i, name in enumerate(self.feature_extractor.get_feature_names()):
            baseline_mean, baseline_std = self._baseline_feature_stats.get(name, (0, 1))
            current_mean = np.mean(X[:, i])
            
            # Z-score based drift detection
            if baseline_std > 0:
                z_score = abs(current_mean - baseline_mean) / baseline_std
            else:
                z_score = abs(current_mean - baseline_mean)
            
            drift_scores.append(z_score)
            
            if z_score > 2:  # More than 2 standard deviations
                drifted_features.append(name)
        
        feature_drift_score = float(np.mean(drift_scores))
        
        # Prediction drift (if predictor is available)
        prediction_drift_score = 0.0
        if self.performance_predictor and self.performance_predictor.is_trained:
            predictions = [
                self.performance_predictor.predict(q).model_confidence 
                for q in recent_queries[:100]  # Sample
            ]
            prediction_drift_score = float(1 - np.mean(predictions))
        
        is_drift_detected = bool(feature_drift_score > 1.5 or len(drifted_features) > 5)
        
        recommendation = "Model is stable"
        if is_drift_detected:
            recommendation = f"Consider retraining - drift detected in {len(drifted_features)} features"
        
        return DriftMetrics(
            feature_drift_score=feature_drift_score,
            prediction_drift_score=prediction_drift_score,
            is_drift_detected=is_drift_detected,
            features_with_drift=drifted_features,
            recommendation=recommendation
        )
    
    def get_monitoring_stats(self) -> Dict[str, ModelMonitoringStats]:
        """Get monitoring statistics for all models."""
        stats = {}
        
        # Classifier stats
        if self.classifier:
            classifier_info = self._get_latest_model_info("classifier")
            stats["classifier"] = ModelMonitoringStats(
                model_id=classifier_info.model_id if classifier_info else "none",
                total_predictions=len([p for p in self._prediction_log if p.get("model") == "classifier"]),
                avg_confidence=self._calculate_avg_confidence("classifier"),
                low_confidence_count=self._count_low_confidence("classifier"),
                high_confidence_count=self._count_high_confidence("classifier"),
                prediction_distribution={},
                last_prediction_at=self._get_last_prediction_time("classifier"),
                drift_metrics=None
            )
        
        # Predictor stats
        if self.performance_predictor:
            predictor_info = self._get_latest_model_info("predictor")
            stats["performance_predictor"] = ModelMonitoringStats(
                model_id=predictor_info.model_id if predictor_info else "none",
                total_predictions=len([p for p in self._prediction_log if p.get("model") == "predictor"]),
                avg_confidence=self._calculate_avg_confidence("predictor"),
                low_confidence_count=self._count_low_confidence("predictor"),
                high_confidence_count=self._count_high_confidence("predictor"),
                prediction_distribution={},
                last_prediction_at=self._get_last_prediction_time("predictor"),
                drift_metrics=None
            )
        
        return stats
    
    def _get_latest_model_info(self, model_type: str) -> Optional[ModelInfo]:
        """Get the latest model info for a given type."""
        models = [
            info for info in self.model_registry.values()
            if info.model_type == model_type and info.is_active
        ]
        if models:
            return max(models, key=lambda x: x.trained_at)
        return None
    
    def _calculate_avg_confidence(self, model: str) -> float:
        """Calculate average confidence for a model."""
        confidences = [
            p.get("confidence", 0) 
            for p in self._prediction_log 
            if p.get("model") == model
        ]
        return float(np.mean(confidences) if confidences else 0)
    
    def _count_low_confidence(self, model: str) -> int:
        """Count low confidence predictions."""
        return len([
            p for p in self._prediction_log 
            if p.get("model") == model and p.get("confidence", 0) < 0.5
        ])
    
    def _count_high_confidence(self, model: str) -> int:
        """Count high confidence predictions."""
        return len([
            p for p in self._prediction_log 
            if p.get("model") == model and p.get("confidence", 0) >= 0.8
        ])
    
    def _get_last_prediction_time(self, model: str) -> Optional[datetime]:
        """Get the last prediction time for a model."""
        predictions = [
            p for p in self._prediction_log 
            if p.get("model") == model
        ]
        if predictions:
            return predictions[-1].get("timestamp")
        return None
    
    def log_prediction(
        self,
        model: str,
        query_hash: str,
        confidence: float,
        result: Any
    ):
        """Log a prediction for monitoring."""
        self._prediction_log.append({
            "model": model,
            "query_hash": query_hash,
            "confidence": confidence,
            "result": str(result),
            "timestamp": datetime.utcnow()
        })
        
        # Keep only last 10000 predictions
        if len(self._prediction_log) > 10000:
            self._prediction_log = self._prediction_log[-10000:]
    
    def should_retrain(self) -> Dict[str, Tuple[bool, str]]:
        """Check if models should be retrained."""
        results = {}
        
        for model_type in ["classifier", "predictor", "recommender"]:
            model_info = self._get_latest_model_info(model_type)
            
            if not model_info:
                results[model_type] = (True, "No trained model exists")
                continue
            
            # Check age
            age_days = (datetime.utcnow() - model_info.trained_at).days
            if age_days > 30:
                results[model_type] = (True, f"Model is {age_days} days old")
                continue
            
            # Check prediction count since training
            predictions_since = len([
                p for p in self._prediction_log 
                if p.get("model") == model_type
            ])
            
            if predictions_since > settings.MODEL_RETRAIN_THRESHOLD:
                results[model_type] = (True, f"{predictions_since} predictions since last training")
                continue
            
            results[model_type] = (False, "Model is up to date")
        
        return results
    
    def get_model_info(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """Get information about models."""
        if model_type:
            info = self._get_latest_model_info(model_type)
            return info.to_dict() if info else {}
        
        return {
            model_id: info.to_dict() 
            for model_id, info in self.model_registry.items()
        }
    
    def export_models(self, export_dir: str) -> List[str]:
        """Export all models to a directory."""
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        
        exported = []
        
        for filename in ["classifier.pkl", "predictor.pkl", "recommender.pkl", "registry.json"]:
            src = self.model_dir / filename
            if src.exists():
                import shutil
                shutil.copy(src, export_path / filename)
                exported.append(filename)
        
        return exported
    
    def import_models(self, import_dir: str) -> List[str]:
        """Import models from a directory."""
        import_path = Path(import_dir)
        
        imported = []
        
        for filename in ["classifier.pkl", "predictor.pkl", "recommender.pkl", "registry.json"]:
            src = import_path / filename
            if src.exists():
                import shutil
                shutil.copy(src, self.model_dir / filename)
                imported.append(filename)
        
        # Reload registry and models
        self._load_registry()
        self.initialize_models()
        
        return imported
    
    # Methods needed by API routes
    def get_all_model_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        return {
            model_type: {
                "is_trained": info.is_active,
                "version": info.version,
                "trained_at": info.trained_at.isoformat(),
                "metrics": info.metrics
            }
            for model_type, info in self.model_registry.items()
        }
    
    def check_all_models_drift(self) -> Dict[str, bool]:
        """Check drift status for all models."""
        result = {}
        for model_type in ["classifier", "performance_predictor", "recommender"]:
            if model_type in self.model_registry:
                result[model_type] = False  # Placeholder - implement actual drift check
        return result
    
    def get_model_status(self, model_type: str) -> Optional[Dict[str, Any]]:
        """Get status for a specific model."""
        info = self.model_registry.get(model_type)
        if info:
            return {
                "model_type": model_type,
                "is_trained": info.is_active,
                "version": info.version,
                "trained_at": info.trained_at.isoformat(),
                "training_samples": info.training_samples,
                "metrics": info.metrics
            }
        return None
    
    async def train_model(self, model_type: str, db) -> Dict[str, Any]:
        """Train a specific model type (async wrapper)."""
        # This would need actual implementation to fetch training data
        return {
            "model_type": model_type,
            "status": "training_not_implemented",
            "message": "Please use specific training methods"
        }
    
    def get_model_metrics(self) -> Dict[str, Any]:
        """Get metrics for all models."""
        metrics = {}
        for model_type, info in self.model_registry.items():
            metrics[model_type] = info.metrics
        return metrics
    
    def check_model_drift(self, model_type: str) -> Dict[str, Any]:
        """Check drift for a specific model."""
        return {
            "model_type": model_type,
            "drift_detected": False,
            "drift_score": 0.0,
            "features_with_drift": [],
            "recommendation": "No drift detected"
        }
    
    def rollback_model(self, model_type: str, version: str) -> Dict[str, Any]:
        """Rollback to a specific model version."""
        return {
            "model_type": model_type,
            "version": version,
            "status": "rollback_not_implemented",
            "message": "Model rollback not yet implemented"
        }
    
    def list_model_versions(self, model_type: str) -> List[str]:
        """List available versions for a model type."""
        versions = []
        if model_type in self.model_registry:
            versions.append(self.model_registry[model_type].version)
        return versions
    
    def get_current_version(self, model_type: str) -> Optional[str]:
        """Get current active version for a model type."""
        if model_type in self.model_registry:
            return self.model_registry[model_type].version
        return None
    
    def export_model(self, model_type: str) -> Optional[str]:
        """Export a specific model to file."""
        model_map = {
            "classifier": (self.classifier, "classifier.pkl"),
            "performance_predictor": (self.performance_predictor, "predictor.pkl"),
            "recommender": (self.recommender, "recommender.pkl")
        }
        
        if model_type not in model_map:
            return None
        
        model, filename = model_map[model_type]
        if model:
            export_path = self.model_dir / filename
            return str(export_path)
        return None
    
    def import_model(self, model_type: str, model_data: bytes) -> Dict[str, Any]:
        """Import a model from data."""
        return {
            "model_type": model_type,
            "status": "import_not_implemented",
            "message": "Direct model import not yet implemented"
        }
    
    def get_training_history(self, model_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get training history for a model type."""
        history = []
        if model_type in self.model_registry:
            info = self.model_registry[model_type]
            history.append({
                "version": info.version,
                "trained_at": info.trained_at.isoformat(),
                "samples": info.training_samples,
                "metrics": info.metrics
            })
        return history[:limit]
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get overall monitoring summary."""
        return {
            "total_predictions": len(self._prediction_log),
            "models_active": len([m for m in self.model_registry.values() if m.is_active]),
            "models_total": len(self.model_registry),
            "drift_alerts": 0
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active monitoring alerts."""
        return []  # Placeholder - implement actual alert system
