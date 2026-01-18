"""
Query-related API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from typing import List, Optional, cast
from datetime import datetime, timedelta
import hashlib

from app.core.database import get_db
from app.models.query import QueryLog, QueryPattern, QueryStatus, QueryComplexity
from app.schemas.query import (
    QueryLogCreate, QueryLogResponse, QueryLogList,
    QueryPatternResponse, QueryPatternList,
    AnalyzeQueryRequest
)
from app.services.query_analyzer import QueryAnalyzerService
from app.services.query_profiler import QueryProfilerService
from app.ml.query_classifier import QueryClassifier
from app.ml.performance_predictor import PerformancePredictor

router = APIRouter(prefix="/queries", tags=["Queries"])

# Initialize services
analyzer_service = QueryAnalyzerService()
classifier = QueryClassifier()
predictor = PerformancePredictor()


@router.get("/", response_model=QueryLogList)
async def list_queries(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[QueryStatus] = None,
    is_slow: Optional[bool] = None,
    query_type: Optional[str] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    List captured queries with filtering and pagination.
    
    - **page**: Page number (starting from 1)
    - **page_size**: Number of items per page
    - **status**: Filter by optimization status
    - **is_slow**: Filter slow queries only
    - **query_type**: Filter by query type (SELECT, INSERT, etc.)
    - **search**: Search in query text
    """
    query = select(QueryLog)
    
    # Apply filters
    if status:
        query = query.where(QueryLog.status == status)
    if is_slow is not None:
        query = query.where(QueryLog.is_slow == is_slow)
    if query_type:
        query = query.where(QueryLog.query_type == query_type.upper())
    if search:
        query = query.where(QueryLog.query_text.ilike(f"%{search}%"))
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination and ordering
    query = query.order_by(desc(QueryLog.created_at))
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    items = result.scalars().all()
    
    return QueryLogList(
        items=[QueryLogResponse.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if total > 0 else 1
    )


@router.get("/slow", response_model=QueryLogList)
async def list_slow_queries(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    min_time_ms: float = Query(1000, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """
    List slow queries exceeding the specified threshold.
    
    - **min_time_ms**: Minimum execution time in milliseconds
    """
    query = select(QueryLog).where(
        QueryLog.execution_time_ms >= min_time_ms
    ).order_by(desc(QueryLog.execution_time_ms))
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    items = result.scalars().all()
    
    return QueryLogList(
        items=[QueryLogResponse.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if total > 0 else 1
    )


@router.get("/frequent", response_model=QueryPatternList)
async def list_frequent_queries(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    min_occurrences: int = Query(5, ge=1),
    db: AsyncSession = Depends(get_db)
):
    """
    List frequently occurring query patterns.
    
    - **min_occurrences**: Minimum number of occurrences
    """
    query = select(QueryPattern).where(
        QueryPattern.occurrence_count >= min_occurrences
    ).order_by(desc(QueryPattern.occurrence_count))
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    items = result.scalars().all()
    
    return QueryPatternList(
        items=[QueryPatternResponse.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if total > 0 else 1
    )


@router.get("/{query_id}", response_model=QueryLogResponse)
async def get_query(
    query_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific query by ID."""
    query_log = await db.get(QueryLog, query_id)
    if not query_log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query {query_id} not found"
        )
    return QueryLogResponse.model_validate(query_log)


@router.post("/", response_model=QueryLogResponse, status_code=status.HTTP_201_CREATED)
async def capture_query(
    query_data: QueryLogCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Capture a new query for analysis.
    
    This endpoint is used to submit queries for optimization analysis.
    """
    # Analyze the query
    analysis = analyzer_service.analyze(query_data.query_text)
    
    # Classify the query
    classification = classifier.classify(query_data.query_text)
    
    # Determine if slow
    is_slow = query_data.execution_time_ms >= 1000  # Default threshold
    
    # Create query log entry
    query_log = QueryLog(
        query_hash=analysis.query_hash,
        query_text=query_data.query_text,
        normalized_query=analysis.normalized_query,
        execution_time_ms=query_data.execution_time_ms,
        rows_examined=query_data.rows_examined,
        rows_returned=query_data.rows_returned,
        execution_plan=query_data.execution_plan,
        query_type=analysis.query_type.value,
        tables_involved=analysis.tables,
        complexity=QueryComplexity(analysis.estimated_complexity),
        status=QueryStatus.PENDING,
        is_slow=is_slow,
        captured_at=datetime.utcnow()
    )
    
    db.add(query_log)
    await db.commit()
    await db.refresh(query_log)
    
    # Update or create pattern
    await _update_query_pattern(db, analysis, query_data.execution_time_ms)
    
    return QueryLogResponse.model_validate(query_log)


@router.post("/analyze")
async def analyze_query(
    request: AnalyzeQueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze a query without capturing it.
    
    Returns detailed analysis including:
    - Query structure
    - Potential issues
    - Classification
    - Performance prediction
    - Optimization suggestions
    """
    # Analyze query structure
    analysis = analyzer_service.analyze(request.query_text)
    
    # Classify the query
    classification = classifier.classify(request.query_text)
    
    # Predict performance
    prediction = predictor.predict(request.query_text)
    
    return {
        "query_hash": analysis.query_hash,
        "analysis": {
            "query_type": analysis.query_type.value,
            "tables": analysis.tables,
            "columns": analysis.columns,
            "joins": analysis.joins,
            "where_conditions": analysis.where_conditions,
            "complexity": analysis.estimated_complexity,
            "has_subquery": analysis.has_subquery,
            "has_aggregation": analysis.has_aggregation,
            "potential_issues": analysis.potential_issues
        },
        "classification": {
            "category": classification.category.value,
            "category_confidence": classification.category_confidence,
            "optimization_priority": classification.optimization_priority.value,
            "priority_confidence": classification.priority_confidence,
            "suggested_optimizations": classification.optimization_types,
            "reasoning": classification.reasoning
        },
        "performance_prediction": {
            "predicted_time_ms": prediction.predicted_time_ms,
            "confidence_interval": prediction.confidence_interval,
            "model_confidence": prediction.model_confidence,
            "performance_class": prediction.performance_class,
            "is_slow_prediction": prediction.is_slow_prediction
        }
    }


@router.get("/{query_id}/explain")
async def explain_query(
    query_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get the execution plan for a specific query."""
    query_log = await db.get(QueryLog, query_id)
    if not query_log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query {query_id} not found"
        )
    
    execution_plan = cast(dict, query_log.execution_plan)
    if execution_plan:
        return {
            "query_id": query_id,
            "execution_plan": execution_plan,
            "cached": True
        }
    
    # If no cached plan, we can't get it without executing
    return {
        "query_id": query_id,
        "execution_plan": None,
        "cached": False,
        "message": "No execution plan available. Submit query with include_plan=True"
    }


@router.delete("/{query_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_query(
    query_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete a query log entry."""
    query_log = await db.get(QueryLog, query_id)
    if not query_log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query {query_id} not found"
        )
    
    await db.delete(query_log)
    await db.commit()


async def _update_query_pattern(
    db: AsyncSession, 
    analysis, 
    execution_time_ms: float
):
    """Update or create a query pattern based on the analysis."""
    # Check if pattern exists
    result = await db.execute(
        select(QueryPattern).where(QueryPattern.pattern_hash == analysis.query_hash)
    )
    pattern = result.scalar_one_or_none()
    
    if pattern:
        # Update existing pattern - use type ignore for SQLAlchemy ORM assignments
        occurrence_count = cast(int, pattern.occurrence_count)
        pattern.occurrence_count = occurrence_count + 1  # type: ignore[assignment]
        pattern.last_seen = datetime.utcnow()  # type: ignore[assignment]
        
        # Update statistics
        avg_exec_time = cast(float, pattern.avg_execution_time_ms)
        if avg_exec_time:
            n = occurrence_count + 1
            old_avg = avg_exec_time
            pattern.avg_execution_time_ms = old_avg + (execution_time_ms - old_avg) / n  # type: ignore[assignment]
        else:
            pattern.avg_execution_time_ms = execution_time_ms  # type: ignore[assignment]
        
        min_exec_time = cast(float, pattern.min_execution_time_ms)
        max_exec_time = cast(float, pattern.max_execution_time_ms)
        if not min_exec_time or execution_time_ms < min_exec_time:
            pattern.min_execution_time_ms = execution_time_ms  # type: ignore[assignment]
        if not max_exec_time or execution_time_ms > max_exec_time:
            pattern.max_execution_time_ms = execution_time_ms  # type: ignore[assignment]
    else:
        # Create new pattern
        pattern = QueryPattern(
            pattern_hash=analysis.query_hash,
            pattern_template=analysis.normalized_query,
            occurrence_count=1,
            avg_execution_time_ms=execution_time_ms,
            min_execution_time_ms=execution_time_ms,
            max_execution_time_ms=execution_time_ms,
            complexity=QueryComplexity(analysis.estimated_complexity),
            tables_involved=analysis.tables,
            joins_count=len(analysis.joins),
            subqueries_count=1 if analysis.has_subquery else 0,
            first_seen=datetime.utcnow(),
            last_seen=datetime.utcnow()
        )
        db.add(pattern)
    
    await db.commit()
