"""
Application configuration settings.
"""
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_NAME: str = "NeuroQO"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    
    # Database Configuration
    DATABASE_TYPE: str = "postgresql"  # postgresql or mysql
    DATABASE_HOST: str = "localhost"
    DATABASE_PORT: int = 5432
    DATABASE_USER: str = "postgres"
    DATABASE_PASSWORD: str = ""
    DATABASE_NAME: str = "neuroqo"
    
    # Target Database (the database being optimized)
    TARGET_DB_HOST: str = "localhost"
    TARGET_DB_PORT: int = 5432
    TARGET_DB_USER: str = "postgres"
    TARGET_DB_PASSWORD: str = ""
    TARGET_DB_NAME: str = "target_db"
    
    # ML Model Settings
    MODEL_PATH: str = "models/"
    MODEL_RETRAIN_THRESHOLD: int = 100  # Retrain after N new queries
    SLOW_QUERY_THRESHOLD_MS: float = 1000.0  # Queries slower than this are "slow"
    
    # Query Analysis Settings
    QUERY_LOG_RETENTION_DAYS: int = 30
    MAX_RECOMMENDATIONS_PER_QUERY: int = 5
    
    # Cache Settings
    CACHE_TTL_SECONDS: int = 3600
    ENABLE_QUERY_CACHE: bool = True
    
    # API Settings
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: str = "*"
    
    # Security Settings
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    @property
    def database_url(self) -> str:
        """Construct the main database URL."""
        if self.DATABASE_TYPE == "postgresql":
            return f"postgresql+asyncpg://{self.DATABASE_USER}:{self.DATABASE_PASSWORD}@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"
        else:
            return f"mysql+aiomysql://{self.DATABASE_USER}:{self.DATABASE_PASSWORD}@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"
    
    @property
    def sync_database_url(self) -> str:
        """Construct synchronous database URL for migrations."""
        if self.DATABASE_TYPE == "postgresql":
            return f"postgresql://{self.DATABASE_USER}:{self.DATABASE_PASSWORD}@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"
        else:
            return f"mysql://{self.DATABASE_USER}:{self.DATABASE_PASSWORD}@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"
    
    @property
    def target_database_url(self) -> str:
        """Construct the target database URL."""
        if self.DATABASE_TYPE == "postgresql":
            return f"postgresql+asyncpg://{self.TARGET_DB_USER}:{self.TARGET_DB_PASSWORD}@{self.TARGET_DB_HOST}:{self.TARGET_DB_PORT}/{self.TARGET_DB_NAME}"
        else:
            return f"mysql+aiomysql://{self.TARGET_DB_USER}:{self.TARGET_DB_PASSWORD}@{self.TARGET_DB_HOST}:{self.TARGET_DB_PORT}/{self.TARGET_DB_NAME}"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
