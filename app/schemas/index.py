"""
Index-related Pydantic schemas.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class IndexType(str, Enum):
    BTREE = "btree"
    HASH = "hash"
    GIN = "gin"
    GIST = "gist"
    BRIN = "brin"
    FULLTEXT = "fulltext"


class IndexStatus(str, Enum):
    RECOMMENDED = "recommended"
    APPROVED = "approved"
    APPLIED = "applied"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class IndexRecommendationResponse(BaseModel):
    """Schema for index recommendation response."""
    id: int
    table_name: str
    column_names: List[str]
    index_type: IndexType
    index_name: Optional[str] = None
    create_statement: str
    drop_statement: Optional[str] = None
    estimated_improvement_ms: Optional[float] = None
    estimated_storage_mb: Optional[float] = None
    affected_queries_count: int
    model_confidence: Optional[float] = None
    reasoning: Optional[str] = None
    status: IndexStatus
    priority: int
    created_at: datetime
    applied_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class IndexRecommendationList(BaseModel):
    """Schema for paginated index recommendation list."""
    items: List[IndexRecommendationResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class IndexActionRequest(BaseModel):
    """Request to apply or rollback an index."""
    recommendation_id: int
    action: str = Field(..., pattern="^(apply|rollback|reject|approve)$")
    reason: Optional[str] = None


class IndexActionResponse(BaseModel):
    """Response after an index action."""
    success: bool
    message: str
    recommendation_id: int
    new_status: IndexStatus
    executed_sql: Optional[str] = None
    error_message: Optional[str] = None


class IndexHistoryResponse(BaseModel):
    """Schema for index history response."""
    id: int
    recommendation_id: Optional[int] = None
    index_name: str
    table_name: str
    action: str
    sql_executed: str
    is_successful: bool
    is_rolled_back: bool
    error_message: Optional[str] = None
    executed_at: datetime
    rolled_back_at: Optional[datetime] = None
    before_metrics: Optional[dict] = None
    after_metrics: Optional[dict] = None
    
    class Config:
        from_attributes = True
