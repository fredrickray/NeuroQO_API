"""
Dashboard API endpoints for monitoring and analytics.
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_
from typing import Optional, cast
from datetime import datetime, timedelta

from app.core.database import get_db
from app.models.query import QueryLog, QueryPattern, OptimizationResult, QueryStatus, QueryComplexity
from app.models.index import IndexRecommendation, IndexStatus
from app.models.experiment import Experiment, ExperimentStatus
from app.ml.model_manager import ModelManager

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

# Initialize model manager
model_manager = ModelManager()


@router.get("/overview")
async def get_dashboard_overview(
    db: AsyncSession = Depends(get_db)
):
    """
    Get overall system health and statistics.
    
    Returns key metrics for the dashboard overview.
    """
    now = datetime.utcnow()
    last_24h = now - timedelta(hours=24)
    last_7d = now - timedelta(days=7)
    
    # Query counts
    total_queries = await db.scalar(select(func.count()).select_from(QueryLog)) or 0
    queries_24h = await db.scalar(
        select(func.count()).select_from(QueryLog).where(QueryLog.captured_at >= last_24h)
    ) or 0
    
    # Slow query counts
    slow_queries = await db.scalar(
        select(func.count()).select_from(QueryLog).where(QueryLog.is_slow == True)
    ) or 0
    slow_queries_24h = await db.scalar(
        select(func.count()).select_from(QueryLog).where(
            QueryLog.is_slow == True,
            QueryLog.captured_at >= last_24h
        )
    ) or 0
    
    # Optimization counts
    total_optimizations = await db.scalar(select(func.count()).select_from(OptimizationResult)) or 0
    applied_optimizations = await db.scalar(
        select(func.count()).select_from(OptimizationResult).where(
            OptimizationResult.is_applied == True
        )
    ) or 0
    
    # Average improvement
    avg_improvement = await db.scalar(
        select(func.avg(OptimizationResult.improvement_percentage)).where(
            OptimizationResult.is_applied == True
        )
    ) or 0
    
    # Active experiments
    active_experiments = await db.scalar(
        select(func.count()).select_from(Experiment).where(
            Experiment.status == ExperimentStatus.RUNNING
        )
    ) or 0
    
    # Pending index recommendations
    pending_indexes = await db.scalar(
        select(func.count()).select_from(IndexRecommendation).where(
            IndexRecommendation.status == IndexStatus.RECOMMENDED
        )
    ) or 0
    
    # Query patterns
    total_patterns = await db.scalar(select(func.count()).select_from(QueryPattern)) or 0
    
    return {
        "timestamp": now.isoformat(),
        "queries": {
            "total": total_queries,
            "last_24h": queries_24h,
            "slow_total": slow_queries,
            "slow_24h": slow_queries_24h,
            "slow_percentage": round(slow_queries / total_queries * 100, 2) if total_queries > 0 else 0
        },
        "optimizations": {
            "total": total_optimizations,
            "applied": applied_optimizations,
            "pending": total_optimizations - applied_optimizations,
            "average_improvement": round(float(avg_improvement), 2)
        },
        "patterns": {
            "total": total_patterns
        },
        "experiments": {
            "active": active_experiments
        },
        "indexes": {
            "pending_recommendations": pending_indexes
        },
        "system_health": "healthy"  # Could add actual health checks
    }


@router.get("/queries/trends")
async def get_query_trends(
    days: int = Query(7, ge=1, le=90),
    db: AsyncSession = Depends(get_db)
):
    """
    Get query volume and performance trends over time.
    """
    now = datetime.utcnow()
    start_date = now - timedelta(days=days)
    
    trends = []
    for i in range(days):
        day_start = start_date + timedelta(days=i)
        day_end = day_start + timedelta(days=1)
        
        # Total queries for the day
        query_count = await db.scalar(
            select(func.count()).select_from(QueryLog).where(
                QueryLog.captured_at >= day_start,
                QueryLog.captured_at < day_end
            )
        )
        
        # Slow queries for the day
        slow_count = await db.scalar(
            select(func.count()).select_from(QueryLog).where(
                QueryLog.captured_at >= day_start,
                QueryLog.captured_at < day_end,
                QueryLog.is_slow == True
            )
        )
        
        # Average execution time
        avg_time = await db.scalar(
            select(func.avg(QueryLog.execution_time_ms)).where(
                QueryLog.captured_at >= day_start,
                QueryLog.captured_at < day_end
            )
        ) or 0
        
        trends.append({
            "date": day_start.strftime("%Y-%m-%d"),
            "total_queries": query_count,
            "slow_queries": slow_count,
            "avg_execution_time_ms": round(avg_time, 2)
        })
    
    return {
        "period_days": days,
        "trends": trends
    }


@router.get("/queries/distribution")
async def get_query_distribution(
    db: AsyncSession = Depends(get_db)
):
    """
    Get distribution of queries by type, complexity, and status.
    """
    # By type
    type_query = select(
        QueryLog.query_type,
        func.count().label("count")
    ).group_by(QueryLog.query_type)
    type_result = await db.execute(type_query)
    by_type = {row.query_type: row.count for row in type_result}
    
    # By complexity
    complexity_query = select(
        QueryLog.complexity,
        func.count().label("count")
    ).group_by(QueryLog.complexity)
    complexity_result = await db.execute(complexity_query)
    by_complexity = {
        row.complexity.value if row.complexity else "UNKNOWN": row.count 
        for row in complexity_result
    }
    
    # By status
    status_query = select(
        QueryLog.status,
        func.count().label("count")
    ).group_by(QueryLog.status)
    status_result = await db.execute(status_query)
    by_status = {row.status.value: row.count for row in status_result}
    
    return {
        "by_type": by_type,
        "by_complexity": by_complexity,
        "by_status": by_status
    }


@router.get("/queries/top-slow")
async def get_top_slow_queries(
    limit: int = Query(10, ge=1, le=50),
    days: int = Query(7, ge=1, le=90),
    db: AsyncSession = Depends(get_db)
):
    """
    Get the slowest queries in the specified period.
    """
    since = datetime.utcnow() - timedelta(days=days)
    
    query = select(QueryLog).where(
        QueryLog.captured_at >= since
    ).order_by(desc(QueryLog.execution_time_ms)).limit(limit)
    
    result = await db.execute(query)
    queries = result.scalars().all()
    
    return {
        "period_days": days,
        "top_slow_queries": [
            {
                "id": q.id,
                "query_type": q.query_type,
                "execution_time_ms": q.execution_time_ms,
                "tables": q.tables_involved,
                "status": q.status.value,
                "captured_at": q.captured_at.isoformat()
            }
            for q in queries
        ]
    }


@router.get("/queries/top-frequent")
async def get_top_frequent_patterns(
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db)
):
    """
    Get the most frequently occurring query patterns.
    """
    query = select(QueryPattern).order_by(
        desc(QueryPattern.occurrence_count)
    ).limit(limit)
    
    result = await db.execute(query)
    patterns = result.scalars().all()
    
    return {
        "top_patterns": [
            {
                "id": p.id,
                "pattern_hash": p.pattern_hash,
                "occurrence_count": p.occurrence_count,
                "avg_execution_time_ms": round(float(cast(float, p.avg_execution_time_ms) or 0), 2),
                "complexity": cast(str, p.complexity) if p.complexity is not None else None,
                "tables": p.tables_involved,
                "is_optimizable": p.is_optimizable
            }
            for p in patterns
        ]
    }


@router.get("/optimizations/impact")
async def get_optimization_impact(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
):
    """
    Get the impact of applied optimizations.
    """
    since = datetime.utcnow() - timedelta(days=days)
    
    # Get applied optimizations
    query = select(OptimizationResult).where(
        OptimizationResult.is_applied == True,
        OptimizationResult.applied_at >= since
    )
    result = await db.execute(query)
    optimizations = result.scalars().all()
    
    if not optimizations:
        return {
            "period_days": days,
            "total_optimizations": 0,
            "impact": None
        }
    
    # Calculate impact
    total_original_time = sum(float(o.original_execution_time_ms) if o.original_execution_time_ms is not None else 0.0 for o in optimizations)  # type: ignore[arg-type]
    total_optimized_time = sum(float(o.optimized_execution_time_ms) if o.optimized_execution_time_ms is not None else 0.0 for o in optimizations)  # type: ignore[arg-type]
    avg_improvement = sum(float(o.improvement_percentage) if o.improvement_percentage is not None else 0.0 for o in optimizations) / len(optimizations)  # type: ignore[arg-type]
    
    # Group by optimization type
    by_type = {}
    for opt in optimizations:
        opt_type = opt.optimization_type or "unknown"
        if opt_type not in by_type:
            by_type[opt_type] = {"count": 0, "total_improvement": 0}
        by_type[opt_type]["count"] += 1
        by_type[opt_type]["total_improvement"] += float(opt.improvement_percentage) if opt.improvement_percentage is not None else 0.0  # type: ignore[arg-type]
    
    for opt_type in by_type:
        by_type[opt_type]["avg_improvement"] = round(
            by_type[opt_type]["total_improvement"] / by_type[opt_type]["count"], 2
        )
    
    return {
        "period_days": days,
        "total_optimizations": len(optimizations),
        "impact": {
            "total_time_saved_ms": max(0, total_original_time - total_optimized_time),
            "avg_improvement_percentage": round(float(avg_improvement), 2),
            "by_type": by_type
        }
    }


@router.get("/models/status")
async def get_model_status():
    """
    Get the status of ML models.
    """
    return {
        "models": model_manager.get_all_model_status(),
        "monitoring": {
            "metrics": model_manager.get_model_metrics(),
            "drift_alerts": model_manager.check_all_models_drift()
        }
    }


@router.get("/models/performance")
async def get_model_performance(
    model_type: Optional[str] = None
):
    """
    Get ML model performance metrics.
    """
    metrics = model_manager.get_model_metrics()
    
    if model_type:
        if model_type in metrics:
            return {model_type: metrics[model_type]}
        return {"error": f"Model type '{model_type}' not found"}
    
    return {"model_metrics": metrics}


@router.get("/alerts")
async def get_system_alerts(
    db: AsyncSession = Depends(get_db)
):
    """
    Get current system alerts and warnings.
    """
    alerts = []
    now = datetime.utcnow()
    last_hour = now - timedelta(hours=1)
    
    # Check for spike in slow queries
    slow_queries_hour = await db.scalar(
        select(func.count()).select_from(QueryLog).where(
            QueryLog.is_slow == True,
            QueryLog.captured_at >= last_hour
        )
    ) or 0
    
    if slow_queries_hour > 100:
        alerts.append({
            "type": "warning",
            "category": "slow_queries",
            "message": f"High number of slow queries in last hour: {slow_queries_hour}",
            "timestamp": now.isoformat()
        })
    
    # Check for model drift
    drift_alerts = model_manager.check_all_models_drift()
    for model_type, has_drift in drift_alerts.items():
        if has_drift:
            alerts.append({
                "type": "info",
                "category": "model_drift",
                "message": f"Model drift detected for {model_type}. Consider retraining.",
                "timestamp": now.isoformat()
            })
    
    # Check for pending recommendations
    pending_count = await db.scalar(
        select(func.count()).select_from(IndexRecommendation).where(
            IndexRecommendation.status == IndexStatus.RECOMMENDED
        )
    ) or 0
    
    if pending_count > 10:
        alerts.append({
            "type": "info",
            "category": "index_recommendations",
            "message": f"{pending_count} pending index recommendations to review",
            "timestamp": now.isoformat()
        })
    
    return {
        "alerts": alerts,
        "count": len(alerts),
        "timestamp": now.isoformat()
    }


@router.get("/comparison")
async def get_before_after_comparison(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
):
    """
    Get before/after comparison for applied optimizations.
    
    Shows the performance improvement from optimizations.
    """
    since = datetime.utcnow() - timedelta(days=days)
    
    query = select(OptimizationResult).where(
        OptimizationResult.is_applied == True,
        OptimizationResult.applied_at >= since,
        OptimizationResult.original_execution_time_ms.isnot(None),
        OptimizationResult.optimized_execution_time_ms.isnot(None)
    )
    
    result = await db.execute(query)
    optimizations = result.scalars().all()
    
    comparisons = []
    for opt in optimizations[:20]:  # Limit to 20 for the chart
        comparisons.append({
            "id": opt.id,
            "optimization_type": opt.optimization_type,
            "before_ms": opt.original_execution_time_ms,
            "after_ms": opt.optimized_execution_time_ms,
            "improvement_pct": opt.improvement_percentage,
            "applied_at": cast(datetime, opt.applied_at).isoformat() if opt.applied_at is not None else None
        })
    
    # Summary statistics
    total_before = sum(c["before_ms"] for c in comparisons if c["before_ms"])
    total_after = sum(c["after_ms"] for c in comparisons if c["after_ms"])
    
    return {
        "period_days": days,
        "comparisons": comparisons,
        "summary": {
            "total_optimizations": len(comparisons),
            "total_time_before_ms": total_before,
            "total_time_after_ms": total_after,
            "total_time_saved_ms": total_before - total_after,
            "average_improvement_pct": round(
                sum(c["improvement_pct"] or 0 for c in comparisons) / len(comparisons), 2
            ) if comparisons else 0
        }
    }
