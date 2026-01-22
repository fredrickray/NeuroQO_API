"""
NeuroQO - AI-Driven Adaptive Query Optimizer

Main FastAPI application entry point.
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging
import time
from datetime import datetime

from app.core.config import settings
from app.core.database import engine, Base
from app.api.routes import queries, optimizations, indexes, experiments, dashboard, auth, validation
from app.api.routes import models as ml_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting NeuroQO application...")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/verified")
    
    yield
    
    # Cleanup
    logger.info("Shutting down NeuroQO application...")
    await engine.dispose()


# Create FastAPI application
app = FastAPI(
    title="NeuroQO",
    description="""
    ## AI-Driven Adaptive Query Optimizer
    
    NeuroQO is an intelligent system that observes, analyzes, and optimizes database queries
    using machine learning techniques.
    
    ### Features
    
    * **Query Explorer** - Browse and analyze captured queries
    * **Slow Query Profiler** - Detect and profile slow-running queries
    * **ML-Based Optimization** - Automatic query rewriting and optimization suggestions
    * **Index Advisor** - Intelligent index recommendations
    * **A/B Experiments** - Test optimization strategies with statistical analysis
    * **Model Monitoring** - Track ML model performance and detect drift
    
    ### API Sections
    
    * `/api/v1/queries` - Query management and analysis
    * `/api/v1/optimizations` - Optimization generation and management
    * `/api/v1/indexes` - Index recommendations and management
    * `/api/v1/experiments` - A/B testing experiments
    * `/api/v1/dashboard` - Monitoring and analytics
    * `/api/v1/models` - ML model management
    * `/api/v1/auth` - Authentication and user management
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing information."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = (time.time() - start_time) * 1000
    
    # Log request
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.2f}ms"
    )
    
    # Add processing time header
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    
    return response


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages."""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error",
            "errors": errors
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.exception(f"Unexpected error: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred",
            "type": type(exc).__name__
        }
    )


# Include API routers
API_PREFIX = "/api/v1"

app.include_router(queries.router, prefix=API_PREFIX)
app.include_router(optimizations.router, prefix=API_PREFIX)
app.include_router(indexes.router, prefix=API_PREFIX)
app.include_router(experiments.router, prefix=API_PREFIX)
app.include_router(dashboard.router, prefix=API_PREFIX)
app.include_router(ml_models.router, prefix=API_PREFIX)
app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(validation.router, prefix=API_PREFIX)


# Root endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "NeuroQO",
        "description": "AI-Driven Adaptive Query Optimizer",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "components": {
            "database": "connected",
            "ml_models": "ready"
        }
    }


@app.get("/api/v1", tags=["API Info"])
async def api_info():
    """API version information."""
    return {
        "version": "v1",
        "endpoints": {
            "queries": "/api/v1/queries",
            "optimizations": "/api/v1/optimizations",
            "indexes": "/api/v1/indexes",
            "experiments": "/api/v1/experiments",
            "dashboard": "/api/v1/dashboard",
            "models": "/api/v1/models",
            "auth": "/api/v1/auth",
            "validation": "/api/v1/validation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
