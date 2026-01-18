"""
User model for authentication and authorization.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Enum as SQLEnum
from sqlalchemy.sql import func
import enum

from app.core.database import Base


class UserRole(str, enum.Enum):
    """User roles for access control."""
    VIEWER = "viewer"  # Can view queries and recommendations
    OPERATOR = "operator"  # Can apply/rollback optimizations
    ADMIN = "admin"  # Full access


class User(Base):
    """User model for the system."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # User info
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    
    # Role
    role = Column(SQLEnum(UserRole), default=UserRole.VIEWER)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    last_login = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<User {self.id}: {self.username}>"
