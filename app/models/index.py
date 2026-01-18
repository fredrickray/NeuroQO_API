"""
Index-related database models.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

from app.core.database import Base


class IndexType(str, enum.Enum):
    """Types of database indexes."""
    BTREE = "btree"
    HASH = "hash"
    GIN = "gin"
    GIST = "gist"
    BRIN = "brin"
    FULLTEXT = "fulltext"


class IndexStatus(str, enum.Enum):
    """Status of index recommendation."""
    RECOMMENDED = "recommended"
    APPROVED = "approved"
    APPLIED = "applied"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class IndexRecommendation(Base):
    """ML-generated index recommendations."""
    __tablename__ = "index_recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Index details
    table_name = Column(String(255), nullable=False, index=True)
    column_names = Column(JSON, nullable=False)  # List of columns
    index_type = Column(SQLEnum(IndexType), default=IndexType.BTREE)
    index_name = Column(String(255))  # Suggested index name
    
    # Creation SQL
    create_statement = Column(Text, nullable=False)
    drop_statement = Column(Text)
    
    # Analysis
    estimated_improvement_ms = Column(Float)
    estimated_storage_mb = Column(Float)
    affected_queries_count = Column(Integer, default=0)
    
    # ML model info
    model_confidence = Column(Float)  # 0-1 confidence score
    reasoning = Column(Text)  # Explanation for the recommendation
    
    # Status
    status = Column(SQLEnum(IndexStatus), default=IndexStatus.RECOMMENDED)
    priority = Column(Integer, default=0)  # Higher = more important
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    applied_at = Column(DateTime, nullable=True)
    
    # Relationships
    history = relationship("IndexHistory", back_populates="recommendation")
    
    def __repr__(self):
        return f"<IndexRecommendation {self.id}: {self.table_name} - {self.status}>"


class IndexHistory(Base):
    """History of index changes for rollback capability."""
    __tablename__ = "index_history"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Reference
    recommendation_id = Column(Integer, ForeignKey("index_recommendations.id"), nullable=True)
    
    # Index details
    index_name = Column(String(255), nullable=False)
    table_name = Column(String(255), nullable=False)
    
    # Action
    action = Column(String(20), nullable=False)  # CREATE, DROP, ALTER
    sql_executed = Column(Text, nullable=False)
    rollback_sql = Column(Text)  # SQL to undo this action
    
    # Before/After metrics
    before_metrics = Column(JSON)  # Query performance before
    after_metrics = Column(JSON)  # Query performance after
    
    # Status
    is_successful = Column(Boolean, default=True)
    is_rolled_back = Column(Boolean, default=False)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    executed_at = Column(DateTime, default=datetime.utcnow)
    rolled_back_at = Column(DateTime, nullable=True)
    
    # Relationships
    recommendation = relationship("IndexRecommendation", back_populates="history")
    
    def __repr__(self):
        return f"<IndexHistory {self.id}: {self.action} {self.index_name}>"
