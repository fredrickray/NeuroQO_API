"""
Query-related database models.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

from app.core.database import Base


class QueryStatus(str, enum.Enum):
    """Status of query optimization."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    OPTIMIZED = "optimized"
    FAILED = "failed"
    SKIPPED = "skipped"


class QueryComplexity(str, enum.Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class QueryLog(Base):
    """Log of all queries captured from the target database."""
    __tablename__ = "query_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Query details
    query_hash = Column(String(64), index=True, nullable=False)  # MD5 hash for deduplication
    query_text = Column(Text, nullable=False)
    normalized_query = Column(Text)  # Query with parameters replaced
    
    # Performance metrics
    execution_time_ms = Column(Float, nullable=False)
    rows_examined = Column(Integer)
    rows_returned = Column(Integer)
    
    # Query plan
    execution_plan = Column(JSON)  # EXPLAIN output
    
    # Classification
    query_type = Column(String(20))  # SELECT, INSERT, UPDATE, DELETE
    tables_involved = Column(JSON)  # List of tables
    complexity = Column(SQLEnum(QueryComplexity), default=QueryComplexity.SIMPLE)
    
    # Status
    status = Column(SQLEnum(QueryStatus), default=QueryStatus.PENDING)
    is_slow = Column(Boolean, default=False)
    
    # Timestamps
    captured_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    pattern_id = Column(Integer, ForeignKey("query_patterns.id"), nullable=True)
    pattern = relationship("QueryPattern", back_populates="query_logs")
    optimizations = relationship("OptimizationResult", back_populates="query_log")
    
    def __repr__(self):
        return f"<QueryLog {self.id}: {self.query_type} - {self.execution_time_ms}ms>"


class QueryPattern(Base):
    """Detected query patterns for ML analysis."""
    __tablename__ = "query_patterns"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Pattern identification
    pattern_hash = Column(String(64), unique=True, index=True)
    pattern_template = Column(Text, nullable=False)  # Normalized query template
    
    # Statistics
    occurrence_count = Column(Integer, default=1)
    avg_execution_time_ms = Column(Float)
    min_execution_time_ms = Column(Float)
    max_execution_time_ms = Column(Float)
    std_execution_time_ms = Column(Float)
    
    # Analysis
    complexity = Column(SQLEnum(QueryComplexity))
    tables_involved = Column(JSON)
    joins_count = Column(Integer, default=0)
    subqueries_count = Column(Integer, default=0)
    
    # ML features (stored for quick access)
    feature_vector = Column(JSON)
    
    # Optimization status
    is_optimizable = Column(Boolean, default=True)
    optimization_priority = Column(Integer, default=0)  # Higher = more priority
    
    # Timestamps
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    query_logs = relationship("QueryLog", back_populates="pattern")
    optimizations = relationship("OptimizationResult", back_populates="pattern")
    
    def __repr__(self):
        return f"<QueryPattern {self.id}: {self.occurrence_count} occurrences>"


class OptimizationResult(Base):
    """Results of query optimization attempts."""
    __tablename__ = "optimization_results"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # References
    query_log_id = Column(Integer, ForeignKey("query_logs.id"), nullable=True)
    pattern_id = Column(Integer, ForeignKey("query_patterns.id"), nullable=True)
    
    # Original query
    original_query = Column(Text, nullable=False)
    original_execution_time_ms = Column(Float)
    original_plan = Column(JSON)
    
    # Optimized query
    optimized_query = Column(Text)
    optimized_execution_time_ms = Column(Float)
    optimized_plan = Column(JSON)
    
    # Optimization details
    optimization_type = Column(String(50))  # rewrite, index, cache, etc.
    optimization_rules_applied = Column(JSON)  # List of rules applied
    
    # Performance improvement
    improvement_percentage = Column(Float)  # Negative means regression
    
    # ML model info
    model_version = Column(String(50))
    model_confidence = Column(Float)  # 0-1 confidence score
    
    # Status
    is_applied = Column(Boolean, default=False)
    is_rolled_back = Column(Boolean, default=False)
    applied_at = Column(DateTime, nullable=True)
    rolled_back_at = Column(DateTime, nullable=True)
    
    # Recommendations
    recommendations = Column(JSON)  # Additional suggestions
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    query_log = relationship("QueryLog", back_populates="optimizations")
    pattern = relationship("QueryPattern", back_populates="optimizations")
    
    def __repr__(self):
        return f"<OptimizationResult {self.id}: {self.improvement_percentage}% improvement>"
