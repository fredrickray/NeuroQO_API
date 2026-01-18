"""
Optimization-related API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from typing import List, Optional, cast
from datetime import datetime, timedelta

from app.core.database import get_db
from app.models.query import QueryLog, QueryPattern, OptimizationResult, QueryStatus
from app.schemas.query import (
    OptimizationResultResponse, OptimizationResultList,
    ApplyOptimizationRequest, RollbackOptimizationRequest
)
from app.services.query_analyzer import QueryAnalyzerService
from app.services.query_rewriter import QueryRewriterService
from app.ml.optimization_recommender import OptimizationRecommender
from app.ml.performance_predictor import PerformancePredictor

router = APIRouter(prefix="/optimizations", tags=["Optimizations"])

# Initialize services
analyzer_service = QueryAnalyzerService()
rewriter_service = QueryRewriterService()
recommender = OptimizationRecommender()
predictor = PerformancePredictor()


@router.get("/", response_model=OptimizationResultList)
async def list_optimizations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    is_applied: Optional[bool] = None,
    min_improvement: Optional[float] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    List optimization results with filtering.
    
    - **is_applied**: Filter by applied status
    - **min_improvement**: Minimum improvement percentage
    """
    query = select(OptimizationResult)
    
    if is_applied is not None:
        query = query.where(OptimizationResult.is_applied == is_applied)
    if min_improvement is not None:
        query = query.where(OptimizationResult.improvement_percentage >= min_improvement)
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination and ordering
    query = query.order_by(desc(OptimizationResult.created_at))
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    items = result.scalars().all()
    
    return OptimizationResultList(
        items=[OptimizationResultResponse.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if total > 0 else 1
    )


@router.get("/pending", response_model=OptimizationResultList)
async def list_pending_optimizations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """List optimizations that haven't been applied yet."""
    query = select(OptimizationResult).where(
        OptimizationResult.is_applied == False,
        OptimizationResult.is_rolled_back == False
    ).order_by(desc(OptimizationResult.improvement_percentage))
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    items = result.scalars().all()
    
    return OptimizationResultList(
        items=[OptimizationResultResponse.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if total > 0 else 1
    )


@router.get("/{optimization_id}", response_model=OptimizationResultResponse)
async def get_optimization(
    optimization_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific optimization result."""
    optimization = await db.get(OptimizationResult, optimization_id)
    if not optimization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Optimization {optimization_id} not found"
        )
    return OptimizationResultResponse.model_validate(optimization)


@router.post("/generate/{query_id}")
async def generate_optimization(
    query_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate optimization suggestions for a specific query.
    
    This analyzes the query and creates optimization recommendations
    using both ML models and rule-based approaches.
    """
    # Get the query
    query_log = await db.get(QueryLog, query_id)
    if not query_log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query {query_id} not found"
        )
    
    # Cast SQLAlchemy Column types to Python types
    query_text = cast(str, query_log.query_text)
    
    # Get ML recommendations
    recommendations = recommender.recommend(query_text)
    
    # Try to rewrite the query
    rewrite_result = rewriter_service.rewrite(query_text)
    
    # Predict improvement
    improvement_prediction = None
    if rewrite_result.rewritten_query != query_text:
        improvement_prediction = predictor.predict_improvement(
            query_text,
            rewrite_result.rewritten_query
        )
    
    # Create optimization result
    optimization = OptimizationResult(
        query_log_id=query_id,
        pattern_id=query_log.pattern_id,
        original_query=query_log.query_text,
        original_execution_time_ms=query_log.execution_time_ms,
        original_plan=query_log.execution_plan,
        optimized_query=rewrite_result.rewritten_query if rewrite_result.rules_applied else None,
        optimization_type="query_rewrite" if rewrite_result.rules_applied else "suggestions_only",
        optimization_rules_applied=[r.value for r in rewrite_result.rules_applied],
        improvement_percentage=improvement_prediction["improvement_percentage"] if improvement_prediction else 0,
        model_version=recommender.model_version,
        model_confidence=recommendations.model_confidence,
        recommendations=[
            {
                "type": s.optimization_type.value,
                "description": s.description,
                "hint": s.implementation_hint,
                "estimated_improvement": s.estimated_improvement_pct,
                "confidence": s.confidence,
                "risk": s.risk_level,
                "priority": s.priority
            }
            for s in recommendations.suggestions
        ]
    )
    
    db.add(optimization)
    
    # Update query status
    query_log.status = QueryStatus.ANALYZING.value  # type: ignore[assignment]
    
    await db.commit()
    await db.refresh(optimization)
    
    return {
        "optimization_id": optimization.id,
        "original_query": query_log.query_text,
        "rewritten_query": rewrite_result.rewritten_query,
        "rules_applied": [r.value for r in rewrite_result.rules_applied],
        "estimated_improvement": rewrite_result.estimated_improvement,
        "confidence": rewrite_result.confidence,
        "warnings": rewrite_result.warnings,
        "ml_recommendations": {
            "suggestions": [
                {
                    "type": s.optimization_type.value,
                    "description": s.description,
                    "hint": s.implementation_hint,
                    "estimated_improvement": s.estimated_improvement_pct,
                    "confidence": s.confidence,
                    "risk": s.risk_level
                }
                for s in recommendations.suggestions
            ],
            "overall_score": recommendations.overall_optimization_score,
            "issues": recommendations.query_issues
        },
        "improvement_prediction": improvement_prediction
    }


@router.post("/apply")
async def apply_optimization(
    request: ApplyOptimizationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Apply an optimization.
    
    This marks the optimization as applied. In a production system,
    this might also update query routing or caching rules.
    """
    optimization = await db.get(OptimizationResult, request.optimization_id)
    if not optimization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Optimization {request.optimization_id} not found"
        )
    
    # Cast SQLAlchemy columns to Python types for conditionals
    is_applied = cast(bool, optimization.is_applied)
    model_confidence_val = cast(float, optimization.model_confidence)
    
    if is_applied:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Optimization is already applied"
        )
    
    # Check confidence threshold
    if not request.force and model_confidence_val and model_confidence_val < 0.5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model confidence ({model_confidence_val:.2f}) is below threshold. Use force=True to override."
        )
    
    # Apply the optimization
    optimization.is_applied = True  # type: ignore[assignment]
    optimization.applied_at = datetime.utcnow()  # type: ignore[assignment]
    
    # Update the query status
    query_log_id = cast(int, optimization.query_log_id)
    if query_log_id:
        query_log = await db.get(QueryLog, query_log_id)
        if query_log:
            query_log.status = QueryStatus.OPTIMIZED.value  # type: ignore[assignment]
    
    await db.commit()
    
    return {
        "success": True,
        "optimization_id": optimization.id,
        "applied_at": optimization.applied_at,
        "message": "Optimization applied successfully"
    }


@router.post("/rollback")
async def rollback_optimization(
    request: RollbackOptimizationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Rollback an applied optimization.
    
    This reverts the optimization and marks it as rolled back.
    """
    optimization = await db.get(OptimizationResult, request.optimization_id)
    if not optimization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Optimization {request.optimization_id} not found"
        )
    
    # Cast SQLAlchemy columns to Python types for conditionals
    is_applied = cast(bool, optimization.is_applied)
    is_rolled_back = cast(bool, optimization.is_rolled_back)
    
    if not is_applied:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Optimization is not applied"
        )
    
    if is_rolled_back:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Optimization is already rolled back"
        )
    
    # Rollback the optimization
    optimization.is_rolled_back = True  # type: ignore[assignment]
    optimization.rolled_back_at = datetime.utcnow()  # type: ignore[assignment]
    
    # Update the query status
    query_log_id = cast(int, optimization.query_log_id)
    if query_log_id:
        query_log = await db.get(QueryLog, query_log_id)
        if query_log:
            query_log.status = QueryStatus.PENDING.value  # type: ignore[assignment]
    
    await db.commit()
    
    return {
        "success": True,
        "optimization_id": optimization.id,
        "rolled_back_at": optimization.rolled_back_at,
        "reason": request.reason,
        "message": "Optimization rolled back successfully"
    }


@router.post("/compare")
async def compare_queries(
    original_query: str,
    optimized_query: str
):
    """
    Compare two queries and predict performance difference.
    
    Useful for validating manual optimizations.
    """
    # Analyze both queries
    original_analysis = analyzer_service.analyze(original_query)
    optimized_analysis = analyzer_service.analyze(optimized_query)
    
    # Compare
    comparison = analyzer_service.compare_queries(original_query, optimized_query)
    
    # Predict improvement
    improvement = predictor.predict_improvement(original_query, optimized_query)
    
    # Validate the rewrite
    validation = rewriter_service.validate_rewrite(original_query, optimized_query)
    
    return {
        "comparison": comparison,
        "improvement_prediction": improvement,
        "validation": validation,
        "original_analysis": {
            "complexity": original_analysis.estimated_complexity,
            "tables": original_analysis.tables,
            "issues": original_analysis.potential_issues
        },
        "optimized_analysis": {
            "complexity": optimized_analysis.estimated_complexity,
            "tables": optimized_analysis.tables,
            "issues": optimized_analysis.potential_issues
        }
    }


@router.get("/statistics/summary")
async def get_optimization_statistics(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
):
    """Get summary statistics for optimizations."""
    since = datetime.utcnow() - timedelta(days=days)
    
    # Total optimizations
    total_query = select(func.count()).select_from(OptimizationResult).where(
        OptimizationResult.created_at >= since
    )
    total_result = await db.execute(total_query)
    total = total_result.scalar() or 0
    
    # Applied optimizations
    applied_query = select(func.count()).select_from(OptimizationResult).where(
        OptimizationResult.created_at >= since,
        OptimizationResult.is_applied == True
    )
    applied_result = await db.execute(applied_query)
    applied = applied_result.scalar() or 0
    
    # Average improvement
    avg_query = select(func.avg(OptimizationResult.improvement_percentage)).where(
        OptimizationResult.created_at >= since,
        OptimizationResult.is_applied == True
    )
    avg_result = await db.execute(avg_query)
    avg_improvement = avg_result.scalar() or 0
    
    # Rolled back
    rollback_query = select(func.count()).select_from(OptimizationResult).where(
        OptimizationResult.created_at >= since,
        OptimizationResult.is_rolled_back == True
    )
    rollback_result = await db.execute(rollback_query)
    rolled_back = rollback_result.scalar() or 0
    
    return {
        "period_days": days,
        "total_optimizations": total,
        "applied_optimizations": applied,
        "pending_optimizations": total - applied - rolled_back,
        "rolled_back_optimizations": rolled_back,
        "average_improvement_percentage": round(float(avg_improvement), 2),
        "application_rate": round(applied / total * 100, 2) if total > 0 else 0
    }
