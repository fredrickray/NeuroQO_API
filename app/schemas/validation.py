"""
Validation-related Pydantic schemas for query testing and verification.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime


class ExecuteQueryRequest(BaseModel):
    """Request to execute a raw SQL query against the target database."""
    query: str = Field(..., description="SQL query to execute")
    limit: int = Field(100, ge=1, le=1000, description="Maximum rows to return (safety limit)")


class ExecuteQueryResponse(BaseModel):
    """Response from executing a query."""
    query: str
    execution_time_ms: float
    row_count: int
    columns: List[str]
    rows: List[dict]
    success: bool
    error: Optional[str] = None


class ValidationTestRequest(BaseModel):
    """Request to run a complete validation test on a query."""
    query: str = Field(..., description="Original SQL query to test")
    run_optimized: bool = Field(True, description="Whether to execute the optimized query")
    limit: int = Field(100, ge=1, le=1000, description="Maximum rows to return")


class ValidationTestResponse(BaseModel):
    """Complete validation test results comparing original and optimized queries."""
    # Original query results
    original_query: str
    original_execution_time_ms: float
    original_row_count: int
    original_rows: List[dict]
    
    # Optimized query results (if applicable)
    optimized_query: Optional[str] = None
    optimized_execution_time_ms: Optional[float] = None
    optimized_row_count: Optional[int] = None
    optimized_rows: Optional[List[dict]] = None
    
    # Optimization details
    optimization_rules: List[str] = []
    recommendations: List[dict] = []
    
    # Comparison metrics
    results_match: bool
    improvement_percentage: float
    improvement_actual_ms: float
    
    # Overall status
    success: bool
    error: Optional[str] = None


class CompareQueriesRequest(BaseModel):
    """Request to compare two queries side by side."""
    query_a: str = Field(..., description="First query (typically the original)")
    query_b: str = Field(..., description="Second query (typically the optimized)")
    limit: int = Field(100, ge=1, le=1000, description="Maximum rows to return")


class CompareQueriesResponse(BaseModel):
    """Results of comparing two queries."""
    # Query A results
    query_a: str
    query_a_time_ms: float
    query_a_rows: int
    query_a_results: List[dict]
    
    # Query B results
    query_b: str
    query_b_time_ms: float
    query_b_rows: int
    query_b_results: List[dict]
    
    # Comparison
    results_match: bool
    improvement_percentage: float
    faster_query: str  # "A", "B", or "equal"
    
    success: bool
    error: Optional[str] = None


class SeedDataRequest(BaseModel):
    """Request to seed test tables with sample data."""
    drop_existing: bool = Field(True, description="Drop existing test tables before creating")
    num_products: int = Field(100, ge=10, le=1000)
    num_customers: int = Field(50, ge=10, le=500)
    num_orders: int = Field(200, ge=20, le=2000)


class SeedDataResponse(BaseModel):
    """Response from seeding test data."""
    success: bool
    tables_created: List[str]
    records_inserted: dict
    message: str
    error: Optional[str] = None


class TableInfo(BaseModel):
    """Information about a test table."""
    name: str
    row_count: int
    columns: List[str]


class ListTablesResponse(BaseModel):
    """Response listing available test tables."""
    tables: List[TableInfo]
    success: bool
    error: Optional[str] = None
