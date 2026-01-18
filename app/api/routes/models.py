"""
ML model management API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, cast
import json

from app.core.database import get_db
from app.core.security import get_current_active_user
from app.models.user import User, UserRole
from app.ml.model_manager import ModelManager
from app.ml.query_classifier import QueryClassifier
from app.ml.performance_predictor import PerformancePredictor
from app.ml.optimization_recommender import OptimizationRecommender

router = APIRouter(prefix="/models", tags=["ML Models"])

# Initialize managers
model_manager = ModelManager()


@router.get("/status")
async def get_models_status():
    """
    Get status of all ML models.
    
    Returns training status, versions, and health metrics.
    """
    return {
        "models": model_manager.get_all_model_status(),
        "health": {
            "overall": "healthy",
            "drift_detected": any(model_manager.check_all_models_drift().values())
        }
    }


@router.get("/{model_type}/status")
async def get_model_status(model_type: str):
    """Get status of a specific model."""
    model_status = model_manager.get_model_status(model_type)
    if not model_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model type '{model_type}' not found"
        )
    return model_status


@router.post("/{model_type}/train")
async def train_model(
    model_type: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger model training.
    
    This starts an async training job for the specified model.
    Requires operator or admin role.
    """
    if current_user.role not in [UserRole.OPERATOR, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operator or admin access required"
        )
    
    valid_types = ["classifier", "predictor", "recommender"]
    if model_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model type. Must be one of: {valid_types}"
        )
    
    # In production, this would queue a background job
    result = await model_manager.train_model(model_type, db)
    
    return {
        "message": f"Training initiated for {model_type}",
        "job_id": result.get("job_id"),
        "status": "training"
    }


@router.get("/{model_type}/metrics")
async def get_model_metrics(model_type: str):
    """Get performance metrics for a specific model."""
    metrics = model_manager.get_model_metrics()
    
    if model_type not in metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No metrics found for model type '{model_type}'"
        )
    
    return {
        "model_type": model_type,
        "metrics": metrics[model_type]
    }


@router.get("/{model_type}/drift")
async def check_model_drift(model_type: str):
    """Check for model drift."""
    drift_status = model_manager.check_model_drift(model_type)
    
    return {
        "model_type": model_type,
        "drift_detected": drift_status.get("drift_detected", False),
        "drift_score": drift_status.get("drift_score", 0.0),
        "recommendation": drift_status.get("recommendation", "Model is stable")
    }


@router.post("/{model_type}/rollback")
async def rollback_model(
    model_type: str,
    version: Optional[str] = None,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Rollback model to a previous version.
    
    If version is not specified, rolls back to the previous version.
    """
    user_role = current_user.get("role", "")
    if user_role not in [UserRole.OPERATOR.value, UserRole.ADMIN.value]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operator or admin access required"
        )
    
    result = model_manager.rollback_model(model_type, version or "previous")
    
    return {
        "message": f"Model rollback attempted",
        "model_type": model_type,
        "result": result
    }


@router.get("/{model_type}/versions")
async def list_model_versions(model_type: str):
    """List all available versions of a model."""
    versions = model_manager.list_model_versions(model_type)
    
    return {
        "model_type": model_type,
        "versions": versions,
        "current_version": model_manager.get_current_version(model_type)
    }


@router.post("/{model_type}/export")
async def export_model(
    model_type: str,
    current_user: User = Depends(get_current_active_user)
):
    """Export a model for external use."""
    if current_user.role not in [UserRole.OPERATOR, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operator or admin access required"
        )
    
    export_path = model_manager.export_model(model_type)
    
    return {
        "message": "Model exported successfully",
        "model_type": model_type,
        "export_path": export_path
    }


@router.post("/{model_type}/import")
async def import_model(
    model_type: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_active_user)
):
    """Import a model from file."""
    user_role = current_user.get("role", "")
    if user_role != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    contents = await file.read()
    result = model_manager.import_model(model_type, contents)
    
    return {
        "message": "Model import attempted",
        "model_type": model_type,
        "result": result
    }


@router.post("/predict/classify")
async def classify_query(query_text: str):
    """
    Classify a query using the ML model.
    
    Returns the predicted category and optimization priority.
    """
    classifier = QueryClassifier()
    result = classifier.classify(query_text)
    
    return {
        "query_text": query_text[:200] + "..." if len(query_text) > 200 else query_text,
        "classification": {
            "category": result.category.value,
            "category_confidence": result.category_confidence,
            "optimization_priority": result.optimization_priority.value,
            "priority_confidence": result.priority_confidence,
            "optimization_types": result.optimization_types,
            "reasoning": result.reasoning
        }
    }


@router.post("/predict/performance")
async def predict_performance(query_text: str):
    """
    Predict query performance using the ML model.
    
    Returns estimated execution time and confidence.
    """
    predictor = PerformancePredictor()
    result = predictor.predict(query_text)
    
    return {
        "query_text": query_text[:200] + "..." if len(query_text) > 200 else query_text,
        "prediction": {
            "predicted_time_ms": result.predicted_time_ms,
            "confidence_interval": result.confidence_interval,
            "model_confidence": result.model_confidence,
            "performance_class": result.performance_class,
            "is_slow_prediction": result.is_slow_prediction,
            "factors": result.contributing_factors
        }
    }


@router.post("/predict/optimize")
async def get_optimization_recommendations(query_text: str):
    """
    Get optimization recommendations for a query.
    
    Uses ML model to suggest optimizations.
    """
    recommender = OptimizationRecommender()
    result = recommender.recommend(query_text)
    
    return {
        "query_text": query_text[:200] + "..." if len(query_text) > 200 else query_text,
        "recommendations": {
            "suggestions": [
                {
                    "type": s.optimization_type.value,
                    "description": s.description,
                    "implementation_hint": s.implementation_hint,
                    "estimated_improvement": s.estimated_improvement_pct,
                    "confidence": s.confidence,
                    "risk_level": s.risk_level,
                    "priority": s.priority
                }
                for s in result.suggestions
            ],
            "overall_score": result.overall_optimization_score,
            "query_issues": result.query_issues,
            "model_confidence": result.model_confidence
        }
    }


@router.get("/training/history")
async def get_training_history(
    model_type: Optional[str] = None,
    limit: int = 20
):
    """Get training history for models."""
    history = model_manager.get_training_history(model_type or "all", limit)
    
    return {
        "training_history": history
    }


@router.get("/monitoring/summary")
async def get_monitoring_summary():
    """Get summary of model monitoring data."""
    return {
        "summary": model_manager.get_monitoring_summary(),
        "alerts": model_manager.get_active_alerts()
    }
