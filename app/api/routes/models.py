"""
ML model management API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.core.database import get_db
from app.core.security import get_current_active_user
from app.models.user import User, UserRole
from app.ml.model_manager import ModelManager
from app.ml.query_classifier import QueryClassifier
from app.ml.performance_predictor import PerformancePredictor
from app.ml.optimization_recommender import OptimizationRecommender

router = APIRouter(prefix="/models", tags=["ML Models"])

model_manager = ModelManager()


# =========================
# Model Registry & Status
# =========================

@router.get("/status")
async def get_models_status():
    """Get status of all registered models."""
    return {
        "models": model_manager.get_model_info(),
        "monitoring": model_manager.get_monitoring_summary(),
    }


@router.get("/{model_type}/status")
async def get_model_status(model_type: str):
    """Get status of a specific model."""
    info = model_manager.get_model_info(model_type)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model type '{model_type}' not found",
        )
    return info


# =========================
# Training
# =========================

@router.post("/{model_type}/train")
async def train_model(
    model_type: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Trigger model training."""
    if current_user.role not in [UserRole.OPERATOR, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operator or admin access required",
        )

    result = await model_manager.train_model(model_type, db)

    if result.get("status") != "trained":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("message", "Training failed"),
        )

    return result


# =========================
# Metrics & Drift
# =========================

@router.get("/{model_type}/metrics")
async def get_model_metrics(model_type: str):
    """Get performance metrics for a model."""
    metrics = model_manager.get_model_metrics(model_type)
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No metrics found for model '{model_type}'",
        )
    return {
        "model_type": model_type,
        "metrics": metrics,
    }


@router.get("/{model_type}/drift")
async def check_model_drift(model_type: str):
    """Check drift for a model."""
    monitoring = model_manager.get_model_monitoring_stats(model_type)
    if not monitoring:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_type}' not found",
        )

    drift = monitoring.drift_metrics
    return {
        "model_type": model_type,
        "drift_detected": drift.is_drift_detected if drift else False,
        "feature_drift_score": drift.feature_drift_score if drift else 0.0,
        "recommendation": drift.recommendation if drift else "No drift data",
    }


# =========================
# Monitoring
# =========================

@router.get("/{model_type}/monitoring")
async def get_model_monitoring(model_type: str):
    """Get detailed monitoring statistics."""
    stats = model_manager.get_model_monitoring_stats(model_type)
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_type}' not found",
        )
    return stats


@router.get("/monitoring/summary")
async def get_monitoring_summary():
    """Get global monitoring summary."""
    return model_manager.get_monitoring_summary()


# =========================
# Predictions
# =========================

@router.post("/predict/classify")
async def classify_query(query_text: str):
    """Classify a query."""
    classifier = QueryClassifier()
    result = classifier.classify(query_text)

    return {
        "query_text": query_text[:200],
        "classification": {
            "category": result.category.value,
            "category_confidence": result.category_confidence,
            "optimization_priority": result.optimization_priority.value,
            "priority_confidence": result.priority_confidence,
            "optimization_types": result.optimization_types,
            "reasoning": result.reasoning,
        },
    }


@router.post("/predict/performance")
async def predict_performance(query_text: str):
    """Predict query performance."""
    predictor = PerformancePredictor()
    result = predictor.predict(query_text)

    return {
        "query_text": query_text[:200],
        "prediction": {
            "predicted_time_ms": result.predicted_time_ms,
            "confidence_interval": result.confidence_interval,
            "model_confidence": result.model_confidence,
            "performance_class": result.performance_class,
            "is_slow_prediction": result.is_slow_prediction,
            "factors": result.contributing_factors,
        },
    }


@router.post("/predict/optimize")
async def get_optimization_recommendations(query_text: str):
    """Get optimization recommendations."""
    recommender = OptimizationRecommender()
    result = recommender.recommend(query_text)

    return {
        "query_text": query_text[:200],
        "recommendations": {
            "overall_score": result.overall_optimization_score,
            "model_confidence": result.model_confidence,
            "query_issues": result.query_issues,
            "suggestions": [
                {
                    "type": s.optimization_type.value,
                    "description": s.description,
                    "implementation_hint": s.implementation_hint,
                    "estimated_improvement": s.estimated_improvement_pct,
                    "confidence": s.confidence,
                    "risk_level": s.risk_level,
                    "priority": s.priority,
                }
                for s in result.suggestions
            ],
        },
    }
