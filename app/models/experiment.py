"""
Experiment-related database models for A/B testing optimizations.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

from app.core.database import Base


class ExperimentStatus(str, enum.Enum):
    """Status of an experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"


class ExperimentType(str, enum.Enum):
    """Types of experiments."""
    QUERY_REWRITE = "query_rewrite"
    INDEX_CHANGE = "index_change"
    CACHE_STRATEGY = "cache_strategy"
    COMBINED = "combined"


class Experiment(Base):
    """A/B experiment for comparing optimization strategies."""
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Experiment info
    name = Column(String(255), nullable=False)
    description = Column(Text)
    experiment_type = Column(SQLEnum(ExperimentType))
    
    # Configuration
    control_config = Column(JSON)  # Control/baseline configuration
    treatment_config = Column(JSON)  # Treatment/optimized configuration
    
    # Target queries/patterns
    target_query_patterns = Column(JSON)  # List of pattern IDs
    
    # Timing
    started_at = Column(DateTime)
    ended_at = Column(DateTime)
    
    # Traffic split and sample size
    traffic_percentage = Column(Integer, default=50)  # % of traffic to treatment
    required_sample_size = Column(Integer, default=1000)  # Required samples
    
    # Status
    status = Column(SQLEnum(ExperimentStatus), default=ExperimentStatus.DRAFT)
    
    # Results summary
    winner = Column(String(20))  # "baseline" or "variant"
    statistical_significance = Column(Float)  # p-value
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    metrics = relationship("ExperimentMetric", back_populates="experiment")
    
    def __repr__(self):
        return f"<Experiment {self.id}: {self.name} - {self.status}>"


class ExperimentMetric(Base):
    """Metrics collected during an experiment."""
    __tablename__ = "experiment_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Reference
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    
    # Variant info
    variant = Column(String(20), nullable=False)  # "baseline" or "variant"
    
    # Query info
    query_pattern_id = Column(Integer)
    query_log_id = Column(Integer)
    
    # Performance metrics
    execution_time_ms = Column(Float)
    rows_examined = Column(Integer)
    rows_returned = Column(Integer)
    cache_hit = Column(Boolean, default=False)
    error_occurred = Column(Boolean, default=False)
    additional_metrics = Column(JSON)  # For extensibility
    
    # Timestamp
    recorded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="metrics")
    
    def __repr__(self):
        return f"<ExperimentMetric {self.id}: {self.variant} - {self.execution_time_ms}ms>"
