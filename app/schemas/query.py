"""
Query-related Pydantic schemas.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime
from enum import Enum


class QueryStatus(str, Enum):
    PENDING = "pending"
    ANALYZING = "analyzing"
    OPTIMIZED = "optimized"
    FAILED = "failed"
    SKIPPED = "skipped"


class QueryComplexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


# Query Log Schemas
class QueryLogBase(BaseModel):
    query_text: str
    execution_time_ms: float
    rows_examined: Optional[int] = None
    rows_returned: Optional[int] = None
    query_type: Optional[str] = None


class QueryLogCreate(QueryLogBase):
    """Schema for creating a new query log."""
    execution_plan: Optional[dict] = None
    tables_involved: Optional[List[str]] = None


class QueryLogResponse(QueryLogBase):
    """Schema for query log response."""
    id: int
    query_hash: str
    normalized_query: Optional[str] = None
    execution_plan: Optional[dict] = None
    tables_involved: Optional[List[str]] = None
    complexity: Optional[QueryComplexity] = None
    status: QueryStatus
    is_slow: bool
    captured_at: datetime
    created_at: datetime
    pattern_id: Optional[int] = None
    
    class Config:
        from_attributes = True


class QueryLogList(BaseModel):
    """Schema for paginated query log list."""
    items: List[QueryLogResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


# Query Pattern Schemas
class QueryPatternResponse(BaseModel):
    """Schema for query pattern response."""
    id: int
    pattern_hash: str
    pattern_template: str
    occurrence_count: int
    avg_execution_time_ms: Optional[float] = None
    min_execution_time_ms: Optional[float] = None
    max_execution_time_ms: Optional[float] = None
    complexity: Optional[QueryComplexity] = None
    tables_involved: Optional[List[str]] = None
    joins_count: int
    subqueries_count: int
    is_optimizable: bool
    optimization_priority: int
    first_seen: datetime
    last_seen: datetime
    
    class Config:
        from_attributes = True


class QueryPatternList(BaseModel):
    """Schema for paginated query pattern list."""
    items: List[QueryPatternResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


# Optimization Result Schemas
class OptimizationResultResponse(BaseModel):
    """Schema for optimization result response."""
    id: int
    query_log_id: Optional[int] = None
    pattern_id: Optional[int] = None
    original_query: str
    original_execution_time_ms: Optional[float] = None
    optimized_query: Optional[str] = None
    optimized_execution_time_ms: Optional[float] = None
    optimization_type: Optional[str] = None
    optimization_rules_applied: Optional[List[str]] = None
    improvement_percentage: Optional[float] = None
    model_version: Optional[str] = None
    model_confidence: Optional[float] = None
    is_applied: bool
    is_rolled_back: bool
    recommendations: Optional[List[dict]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class OptimizationResultList(BaseModel):
    """Schema for paginated optimization result list."""
    items: List[OptimizationResultResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


# Request schemas for optimization actions
class ApplyOptimizationRequest(BaseModel):
    """Request to apply an optimization."""
    optimization_id: int
    force: bool = False  # Apply even if confidence is low


class RollbackOptimizationRequest(BaseModel):
    """Request to rollback an optimization."""
    optimization_id: int
    reason: Optional[str] = None


class AnalyzeQueryRequest(BaseModel):
    """Request to analyze a specific query."""
    query_text: str
    include_plan: bool = True
    suggest_optimizations: bool = True
