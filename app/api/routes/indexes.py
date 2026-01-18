"""
Index-related API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from typing import List, Optional, cast
from datetime import datetime

from app.core.database import get_db, get_target_db
from app.models.index import IndexRecommendation, IndexHistory, IndexStatus as ModelIndexStatus, IndexType
from app.schemas.index import (
    IndexRecommendationResponse, IndexRecommendationList,
    IndexActionRequest, IndexActionResponse, IndexHistoryResponse,
    IndexStatus as SchemaIndexStatus
)
from app.services.index_advisor import IndexAdvisorService
from app.services.query_analyzer import QueryAnalyzerService

router = APIRouter(prefix="/indexes", tags=["Indexes"])

# Initialize services
index_advisor = IndexAdvisorService()
analyzer = QueryAnalyzerService()


@router.get("/recommendations", response_model=IndexRecommendationList)
async def list_index_recommendations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: Optional[ModelIndexStatus] = Query(None, alias="status"),
    table_name: Optional[str] = None,
    min_confidence: Optional[float] = Query(None, ge=0, le=1),
    db: AsyncSession = Depends(get_db)
):
    """
    List index recommendations with filtering.
    
    - **status**: Filter by recommendation status
    - **table_name**: Filter by table name
    - **min_confidence**: Minimum model confidence
    """
    query = select(IndexRecommendation)
    
    if status_filter:
        query = query.where(IndexRecommendation.status == status_filter)
    if table_name:
        query = query.where(IndexRecommendation.table_name == table_name)
    if min_confidence is not None:
        query = query.where(IndexRecommendation.model_confidence >= min_confidence)
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination and ordering
    query = query.order_by(desc(IndexRecommendation.priority), desc(IndexRecommendation.created_at))
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    items = result.scalars().all()
    
    return IndexRecommendationList(
        items=[IndexRecommendationResponse.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if total > 0 else 1
    )


@router.get("/recommendations/{recommendation_id}", response_model=IndexRecommendationResponse)
async def get_index_recommendation(
    recommendation_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific index recommendation."""
    recommendation = await db.get(IndexRecommendation, recommendation_id)
    if not recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index recommendation {recommendation_id} not found"
        )
    return IndexRecommendationResponse.model_validate(recommendation)


@router.post("/analyze")
async def analyze_for_indexes(
    query_text: str,
    db: AsyncSession = Depends(get_db),
    target_db: AsyncSession = Depends(get_target_db)
):
    """
    Analyze a query and generate index recommendations.
    
    This uses ML-based analysis to suggest indexes that could
    improve the query's performance.
    """
    # Analyze the query
    analysis = analyzer.analyze(query_text)
    
    # Get index recommendations
    recommendations = await index_advisor.analyze_query_for_indexes(
        target_db,
        query_text
    )
    
    # Store recommendations in database
    stored_recommendations = []
    for rec in recommendations:
        # Check if similar recommendation exists
        existing_query = select(IndexRecommendation).where(
            IndexRecommendation.table_name == rec.table_name,
            IndexRecommendation.column_names == rec.column_names,
            IndexRecommendation.status.in_([ModelIndexStatus.RECOMMENDED, ModelIndexStatus.APPROVED])
        )
        existing_result = await db.execute(existing_query)
        existing = existing_result.scalar_one_or_none()
        
        if existing:
            # Update priority if new recommendation is higher - use type ignores for SQLAlchemy
            existing_priority = cast(int, existing.priority)
            existing_affected = cast(int, existing.affected_queries_count)
            if rec.priority > existing_priority:
                existing.priority = rec.priority  # type: ignore[assignment]
                existing.affected_queries_count = existing_affected + 1  # type: ignore[assignment]
            stored_recommendations.append(existing)
        else:
            # Create new recommendation
            new_rec = IndexRecommendation(
                table_name=rec.table_name,
                column_names=rec.column_names,
                index_type=IndexType(rec.index_type),
                index_name=rec.index_name,
                create_statement=rec.create_statement,
                drop_statement=rec.drop_statement,
                estimated_improvement_ms=rec.estimated_improvement_ms,
                estimated_storage_mb=rec.estimated_storage_mb,
                affected_queries_count=1,
                model_confidence=rec.confidence,
                reasoning=rec.reasoning,
                status=ModelIndexStatus.RECOMMENDED,
                priority=rec.priority
            )
            db.add(new_rec)
            stored_recommendations.append(new_rec)
    
    await db.commit()
    
    return {
        "query_analysis": {
            "tables": analysis.tables,
            "where_conditions": analysis.where_conditions,
            "joins": analysis.joins,
            "order_by": analysis.order_by
        },
        "recommendations": [
            {
                "table": rec.table_name,
                "columns": rec.column_names,
                "index_type": rec.index_type,
                "create_sql": rec.create_statement,
                "reasoning": rec.reasoning,
                "priority": rec.priority,
                "confidence": rec.confidence
            }
            for rec in recommendations
        ],
        "recommendation_count": len(recommendations)
    }


@router.post("/action", response_model=IndexActionResponse)
async def perform_index_action(
    request: IndexActionRequest,
    db: AsyncSession = Depends(get_db),
    target_db: AsyncSession = Depends(get_target_db)
):
    """
    Perform an action on an index recommendation.
    
    Actions:
    - **apply**: Create the recommended index
    - **rollback**: Drop a previously created index
    - **approve**: Mark recommendation as approved
    - **reject**: Reject the recommendation
    """
    recommendation = await db.get(IndexRecommendation, request.recommendation_id)
    if not recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index recommendation {request.recommendation_id} not found"
        )
    
    executed_sql: Optional[str] = None
    error_message = None
    rec_status = cast(str, recommendation.status)
    new_status = rec_status
    
    try:
        if request.action == "apply":
            if rec_status == ModelIndexStatus.APPLIED.value:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Index is already applied"
                )
            
            # Execute CREATE INDEX
            from sqlalchemy import text
            create_stmt = cast(str, recommendation.create_statement)
            await target_db.execute(text(create_stmt))
            await target_db.commit()
            
            executed_sql = create_stmt
            new_status = ModelIndexStatus.APPLIED.value
            recommendation.status = new_status  # type: ignore[assignment]
            recommendation.applied_at = datetime.utcnow()  # type: ignore[assignment]
            
            # Record history
            rec_id = cast(int, recommendation.id)
            rec_index_name = cast(str, recommendation.index_name)
            rec_table_name = cast(str, recommendation.table_name)
            drop_stmt = cast(str, recommendation.drop_statement)
            
            history = IndexHistory(
                recommendation_id=rec_id,
                index_name=rec_index_name,
                table_name=rec_table_name,
                action="CREATE",
                sql_executed=create_stmt,
                rollback_sql=drop_stmt,
                is_successful=True,
                executed_at=datetime.utcnow()
            )
            db.add(history)
            
        elif request.action == "rollback":
            if rec_status != ModelIndexStatus.APPLIED.value:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Index is not applied"
                )
            
            # Execute DROP INDEX
            from sqlalchemy import text
            drop_stmt = cast(str, recommendation.drop_statement)
            await target_db.execute(text(drop_stmt))
            await target_db.commit()
            
            executed_sql = drop_stmt
            new_status = ModelIndexStatus.ROLLED_BACK.value
            recommendation.status = new_status  # type: ignore[assignment]
            
            # Record history
            rec_id = cast(int, recommendation.id)
            rec_index_name = cast(str, recommendation.index_name)
            rec_table_name = cast(str, recommendation.table_name)
            
            history = IndexHistory(
                recommendation_id=rec_id,
                index_name=rec_index_name,
                table_name=rec_table_name,
                action="DROP",
                sql_executed=drop_stmt,
                is_successful=True,
                executed_at=datetime.utcnow()
            )
            db.add(history)
            
        elif request.action == "approve":
            new_status = ModelIndexStatus.APPROVED.value
            recommendation.status = new_status  # type: ignore[assignment]
            
        elif request.action == "reject":
            new_status = ModelIndexStatus.REJECTED.value
            recommendation.status = new_status  # type: ignore[assignment]
        
        await db.commit()
        
        rec_id = cast(int, recommendation.id)
        return IndexActionResponse(
            success=True,
            message=f"Action '{request.action}' completed successfully",
            recommendation_id=rec_id,
            new_status=SchemaIndexStatus(new_status),
            executed_sql=executed_sql
        )
        
    except Exception as e:
        error_message = str(e)
        
        # Record failed action in history
        if request.action in ["apply", "rollback"]:
            rec_id = cast(int, recommendation.id)
            rec_index_name = cast(str, recommendation.index_name)
            rec_table_name = cast(str, recommendation.table_name)
            create_stmt = cast(str, recommendation.create_statement)
            drop_stmt = cast(str, recommendation.drop_statement)
            
            history = IndexHistory(
                recommendation_id=rec_id,
                index_name=rec_index_name,
                table_name=rec_table_name,
                action="CREATE" if request.action == "apply" else "DROP",
                sql_executed=create_stmt if request.action == "apply" else drop_stmt,
                is_successful=False,
                error_message=error_message,
                executed_at=datetime.utcnow()
            )
            db.add(history)
            await db.commit()
        
        rec_id = cast(int, recommendation.id)
        rec_status = SchemaIndexStatus(cast(str, recommendation.status))
        return IndexActionResponse(
            success=False,
            message=f"Action '{request.action}' failed",
            recommendation_id=rec_id,
            new_status=rec_status,
            error_message=error_message
        )


@router.get("/existing")
async def list_existing_indexes(
    table_name: Optional[str] = None,
    target_db: AsyncSession = Depends(get_target_db)
):
    """List existing indexes in the target database."""
    indexes = await index_advisor.get_existing_indexes(target_db, table_name or "")
    
    return {
        "indexes": [
            {
                "index_name": idx.index_name,
                "table_name": idx.table_name,
                "columns": idx.columns,
                "index_type": idx.index_type,
                "is_unique": idx.is_unique,
                "is_primary": idx.is_primary,
                "usage_count": idx.usage_count,
                "size_mb": round(idx.size_mb, 2),
                "is_unused": idx.is_unused,
                "is_redundant": idx.is_redundant,
                "recommendations": idx.recommendations
            }
            for idx in indexes
        ],
        "total": len(indexes)
    }


@router.get("/unused")
async def list_unused_indexes(
    min_days_unused: int = Query(30, ge=1),
    target_db: AsyncSession = Depends(get_target_db)
):
    """List indexes that haven't been used recently."""
    unused = await index_advisor.find_unused_indexes(target_db, min_days_unused)
    
    return {
        "unused_indexes": [
            {
                "index_name": idx.index_name,
                "table_name": idx.table_name,
                "columns": idx.columns,
                "size_mb": round(idx.size_mb, 2),
                "recommendations": idx.recommendations
            }
            for idx in unused
        ],
        "total": len(unused),
        "potential_space_savings_mb": round(sum(idx.size_mb for idx in unused), 2)
    }


@router.get("/redundant")
async def list_redundant_indexes(
    target_db: AsyncSession = Depends(get_target_db)
):
    """List redundant indexes (covered by other indexes)."""
    redundant = await index_advisor.find_redundant_indexes(target_db)
    
    return {
        "redundant_indexes": [
            {
                "index_name": idx.index_name,
                "table_name": idx.table_name,
                "columns": idx.columns,
                "size_mb": round(idx.size_mb, 2),
                "recommendations": idx.recommendations
            }
            for idx in redundant
        ],
        "total": len(redundant),
        "potential_space_savings_mb": round(sum(idx.size_mb for idx in redundant), 2)
    }


@router.get("/history")
async def list_index_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    table_name: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """List index action history."""
    query = select(IndexHistory)
    
    if table_name:
        query = query.where(IndexHistory.table_name == table_name)
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination and ordering
    query = query.order_by(desc(IndexHistory.executed_at))
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    items = result.scalars().all()
    
    return {
        "items": [IndexHistoryResponse.model_validate(item) for item in items],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if total > 0 else 1
    }
