"""
Authentication API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta
from typing import Optional, cast

from app.core.database import get_db
from app.core.security import (
    get_password_hash, verify_password, 
    create_access_token, get_current_user, get_current_active_user
)
from app.core.config import settings
from app.models.user import User, UserRole
from app.schemas.user import (
    UserCreate, UserResponse, UserUpdate,
    Token, TokenData
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user.
    
    - **email**: User's email address (must be unique)
    - **username**: Username (must be unique)
    - **password**: Password (min 8 characters)
    """
    # Check if email already exists
    email_query = select(User).where(User.email == user_data.email)
    existing_email = await db.execute(email_query)
    if existing_email.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if username already exists
    username_query = select(User).where(User.username == user_data.username)
    existing_username = await db.execute(username_query)
    if existing_username.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create new user
    user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=get_password_hash(user_data.password),
        full_name=user_data.full_name,
        role=UserRole.VIEWER  # Default role
    )
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return UserResponse.model_validate(user)


@router.post("/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    OAuth2 compatible token login.
    
    Returns an access token for the user.
    """
    # Find user by username or email
    query = select(User).where(
        (User.username == form_data.username) | (User.email == form_data.username)
    )
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(form_data.password, cast(str, user.hashed_password)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if not cast(bool, user.is_active):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()  # type: ignore[assignment]
    await db.commit()
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": cast(str, user.username), "role": cast(UserRole, user.role).value},
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user information."""
    return UserResponse.model_validate(current_user)


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update current user information."""
    # Update fields if provided
    if user_update.full_name is not None:
        current_user.full_name = user_update.full_name  # type: ignore[assignment]
    
    if user_update.email is not None:
        # Check if email already exists
        email_query = select(User).where(
            User.email == user_update.email,
            User.id != current_user.id
        )
        existing = await db.execute(email_query)
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use"
            )
        current_user.email = user_update.email  # type: ignore[assignment]
    
    current_user.updated_at = datetime.utcnow()  # type: ignore[assignment]
    await db.commit()
    await db.refresh(current_user)
    
    return UserResponse.model_validate(current_user)


@router.post("/change-password")
async def change_password(
    current_password: str,
    new_password: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Change current user's password."""
    if not verify_password(current_password, cast(str, current_user.hashed_password)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    if len(new_password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters"
        )
    
    current_user.hashed_password = get_password_hash(new_password)  # type: ignore[assignment]
    current_user.updated_at = datetime.utcnow()  # type: ignore[assignment]
    await db.commit()
    
    return {"message": "Password changed successfully"}


@router.post("/refresh", response_model=Token)
async def refresh_token(
    current_user: User = Depends(get_current_active_user)
):
    """Refresh the access token."""
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": cast(str, current_user.username), "role": cast(UserRole, current_user.role).value},
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """List all users (admin only)."""
    if cast(str, current_user.role) != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    query = select(User).order_by(User.created_at)
    result = await db.execute(query)
    users = result.scalars().all()
    
    return [UserResponse.model_validate(user) for user in users]


@router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: int,
    role: UserRole,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a user's role (admin only)."""
    if cast(str, current_user.role) != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )
    
    if cast(int, user.id) == cast(int, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own role"
        )
    
    user.role = role  # type: ignore[assignment]
    user.updated_at = datetime.utcnow()  # type: ignore[assignment]
    await db.commit()
    
    return {"message": f"User role updated to {role.value}"}


@router.put("/users/{user_id}/activate")
async def toggle_user_active(
    user_id: int,
    is_active: bool,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Activate or deactivate a user (admin only)."""
    if cast(str, current_user.role) != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )
    
    if cast(int, user.id) == cast(int, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate yourself"
        )
    
    user.is_active = is_active  # type: ignore[assignment]
    user.updated_at = datetime.utcnow()  # type: ignore[assignment]
    await db.commit()
    
    status_text = "activated" if is_active else "deactivated"
    return {"message": f"User {status_text} successfully"}
