"""
Experiment-related Pydantic schemas.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"


class ExperimentType(str, Enum):
    QUERY_REWRITE = "query_rewrite"
    INDEX_CHANGE = "index_change"
    CACHE_STRATEGY = "cache_strategy"
    COMBINED = "combined"


class ExperimentCreate(BaseModel):
    """Schema for creating a new experiment."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    experiment_type: ExperimentType
    control_config: dict
    treatment_config: dict
    target_query_patterns: Optional[List[int]] = None
    traffic_percentage: int = Field(default=50, ge=10, le=90)
    required_sample_size: int = Field(default=1000, ge=100)


class ExperimentResponse(BaseModel):
    """Schema for experiment response."""
    id: int
    name: str
    description: Optional[str] = None
    experiment_type: ExperimentType
    control_config: dict
    treatment_config: dict
    target_query_patterns: Optional[List[int]] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    traffic_percentage: int
    required_sample_size: int
    status: ExperimentStatus
    winner: Optional[str] = None
    statistical_significance: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ExperimentList(BaseModel):
    """Schema for paginated experiment list."""
    items: List[ExperimentResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class ExperimentMetricCreate(BaseModel):
    """Schema for creating an experiment metric."""
    variant: str = Field(..., pattern="^(control|treatment)$")
    query_pattern_id: Optional[int] = None
    query_log_id: Optional[int] = None
    execution_time_ms: float = Field(..., ge=0)
    rows_examined: Optional[int] = Field(None, ge=0)
    rows_returned: Optional[int] = Field(None, ge=0)
    cache_hit: bool = False
    error_occurred: bool = False
    additional_metrics: Optional[Dict[str, Any]] = None


class ExperimentMetricResponse(BaseModel):
    """Schema for experiment metric response."""
    id: int
    experiment_id: int
    variant: str
    query_pattern_id: Optional[int] = None
    query_log_id: Optional[int] = None
    execution_time_ms: Optional[float] = None
    rows_examined: Optional[int] = None
    rows_returned: Optional[int] = None
    cache_hit: Optional[bool] = None
    error_occurred: Optional[bool] = None
    additional_metrics: Optional[Dict[str, Any]] = None
    recorded_at: datetime
    
    class Config:
        from_attributes = True


class ExperimentResultResponse(BaseModel):
    """Schema for experiment analysis results."""
    experiment_id: int
    experiment_name: str
    status: str
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    control_metrics: Dict[str, float]
    treatment_metrics: Dict[str, float]
    statistical_analysis: Dict[str, Any]
    recommendation: str
    confidence_level: float
    sample_sizes: Dict[str, int]


class ExperimentMetricsSummary(BaseModel):
    """Summary of experiment metrics."""
    experiment_id: int
    baseline_avg_time_ms: float
    variant_avg_time_ms: float
    baseline_p50_ms: float
    variant_p50_ms: float
    baseline_p95_ms: float
    variant_p95_ms: float
    baseline_sample_count: int
    variant_sample_count: int
    improvement_percentage: float
    is_statistically_significant: bool
    p_value: Optional[float] = None


class ExperimentActionRequest(BaseModel):
    """Request to start/stop an experiment."""
    experiment_id: int
    action: str = Field(..., pattern="^(start|stop|abort)$")
