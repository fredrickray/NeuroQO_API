"""
Dashboard-related Pydantic schemas.
"""
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class SlowQuerySummary(BaseModel):
    """Summary of slow queries."""
    query_hash: str
    query_preview: str  # First 100 chars
    occurrence_count: int
    avg_execution_time_ms: float
    max_execution_time_ms: float
    tables_involved: List[str]
    is_optimized: bool
    potential_improvement_ms: Optional[float] = None


class PerformanceMetrics(BaseModel):
    """Performance metrics over time."""
    timestamp: datetime
    avg_query_time_ms: float
    p50_query_time_ms: float
    p95_query_time_ms: float
    p99_query_time_ms: float
    total_queries: int
    slow_queries: int
    optimized_queries: int


class DashboardStats(BaseModel):
    """Overall dashboard statistics."""
    # Query stats
    total_queries_analyzed: int
    total_slow_queries: int
    total_patterns_detected: int
    
    # Optimization stats
    total_optimizations: int
    applied_optimizations: int
    pending_optimizations: int
    avg_improvement_percentage: float
    
    # Index stats
    total_index_recommendations: int
    applied_indexes: int
    pending_indexes: int
    
    # Model stats
    model_version: str
    model_accuracy: float
    model_last_trained: Optional[datetime] = None
    queries_since_last_train: int
    
    # Performance comparison
    avg_query_time_before_ms: float
    avg_query_time_after_ms: float
    overall_improvement_percentage: float
    
    # Recent activity
    queries_last_24h: int
    optimizations_last_24h: int


class TimeSeriesDataPoint(BaseModel):
    """Single data point for time series charts."""
    timestamp: datetime
    value: float
    label: Optional[str] = None


class PerformanceComparisonChart(BaseModel):
    """Data for before/after performance comparison charts."""
    query_id: int
    query_preview: str
    before_time_ms: float
    after_time_ms: float
    improvement_percentage: float
    optimization_type: str


class ModelMonitoringStats(BaseModel):
    """ML model monitoring statistics."""
    model_version: str
    model_type: str  # "random_forest", "lightgbm", etc.
    
    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Confidence distribution
    avg_confidence: float
    low_confidence_predictions: int  # < 0.5
    high_confidence_predictions: int  # >= 0.8
    
    # Drift detection
    feature_drift_detected: bool
    prediction_drift_detected: bool
    drift_score: Optional[float] = None
    
    # Training info
    training_samples: int
    last_trained: datetime
    next_retrain_threshold: int
    queries_until_retrain: int
