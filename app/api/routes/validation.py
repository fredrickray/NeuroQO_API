"""
Query Validation API endpoints for testing and verifying optimizations.

This module provides endpoints to:
- Execute queries against the target database
- Run validation tests comparing original and optimized queries
- Seed test tables with sample data for testing
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List, Optional
import time
import random
from datetime import datetime, timedelta

from app.core.database import get_target_db
from app.schemas.validation import (
    ExecuteQueryRequest, ExecuteQueryResponse,
    ValidationTestRequest, ValidationTestResponse,
    CompareQueriesRequest, CompareQueriesResponse,
    SeedDataRequest, SeedDataResponse,
    ListTablesResponse, TableInfo
)
from app.services.query_rewriter import QueryRewriterService
from app.ml.optimization_recommender import OptimizationRecommender

router = APIRouter(prefix="/validation", tags=["Validation & Testing"])

# Initialize services
rewriter_service = QueryRewriterService()
recommender = OptimizationRecommender()


async def execute_query_with_timing(
    session: AsyncSession, 
    query: str, 
    limit: int = 100
) -> dict:
    """Execute a query and return results with timing information."""
    try:
        # Add LIMIT if not present for safety
        query_upper = query.upper().strip()
        if query_upper.startswith("SELECT") and "LIMIT" not in query_upper:
            query = f"{query.rstrip(';')} LIMIT {limit}"
        
        start_time = time.perf_counter()
        result = await session.execute(text(query))
        execution_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        rows = result.fetchall()
        columns = list(result.keys()) if rows else []
        
        # Convert rows to dicts
        row_dicts = [dict(zip(columns, row)) for row in rows]
        
        return {
            "success": True,
            "execution_time_ms": round(execution_time, 3),
            "row_count": len(row_dicts),
            "columns": columns,
            "rows": row_dicts,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "execution_time_ms": 0,
            "row_count": 0,
            "columns": [],
            "rows": [],
            "error": str(e)
        }


def compare_results(rows_a: List[dict], rows_b: List[dict]) -> bool:
    """Compare two result sets for equality."""
    if len(rows_a) != len(rows_b):
        return False
    
    # Sort both lists by their dict representations for comparison
    try:
        sorted_a = sorted([str(sorted(r.items())) for r in rows_a])
        sorted_b = sorted([str(sorted(r.items())) for r in rows_b])
        return sorted_a == sorted_b
    except Exception:
        # If sorting fails, try direct comparison
        return rows_a == rows_b


@router.post("/execute", response_model=ExecuteQueryResponse)
async def execute_query(
    request: ExecuteQueryRequest,
    db: AsyncSession = Depends(get_target_db)
):
    """
    Execute a raw SQL query against the target database.
    
    Returns the results with execution time measurements.
    Use this to test individual queries.
    
    **Note**: A LIMIT clause will be added automatically for SELECT queries
    if not present, for safety.
    """
    result = await execute_query_with_timing(db, request.query, request.limit)
    
    return ExecuteQueryResponse(
        query=request.query,
        execution_time_ms=result["execution_time_ms"],
        row_count=result["row_count"],
        columns=result["columns"],
        rows=result["rows"],
        success=result["success"],
        error=result["error"]
    )


@router.post("/test", response_model=ValidationTestResponse)
async def run_validation_test(
    request: ValidationTestRequest,
    db: AsyncSession = Depends(get_target_db)
):
    """
    Run a complete validation test on a query.
    
    This will:
    1. Execute the original query and measure performance
    2. Generate optimization recommendations using the ML system
    3. If an optimized query is produced, execute it
    4. Compare results to verify semantic equivalence
    5. Calculate actual performance improvement
    
    This is the main endpoint for validating that optimizations work correctly.
    """
    # Step 1: Execute original query
    original_result = await execute_query_with_timing(db, request.query, request.limit)
    
    if not original_result["success"]:
        return ValidationTestResponse(
            original_query=request.query,
            original_execution_time_ms=0,
            original_row_count=0,
            original_rows=[],
            results_match=False,
            improvement_percentage=0,
            improvement_actual_ms=0,
            success=False,
            error=f"Failed to execute original query: {original_result['error']}"
        )
    
    # Step 2: Get optimization recommendations
    recommendations = recommender.recommend(request.query)
    
    # Step 3: Try to rewrite the query
    rewrite_result = rewriter_service.rewrite(request.query)
    
    optimized_query = None
    optimized_result = None
    optimization_rules = []
    results_match = True
    improvement_pct = 0.0
    improvement_ms = 0.0
    
    if rewrite_result.rules_applied and rewrite_result.rewritten_query != request.query:
        optimized_query = rewrite_result.rewritten_query
        optimization_rules = [r.value for r in rewrite_result.rules_applied]
        
        # Step 4: Execute optimized query if requested
        if request.run_optimized:
            optimized_result = await execute_query_with_timing(db, optimized_query, request.limit)
            
            if optimized_result["success"]:
                # Step 5: Compare results
                results_match = compare_results(
                    original_result["rows"], 
                    optimized_result["rows"]
                )
                
                # Calculate improvement
                original_time = original_result["execution_time_ms"]
                optimized_time = optimized_result["execution_time_ms"]
                
                if original_time > 0:
                    improvement_pct = ((original_time - optimized_time) / original_time) * 100
                    improvement_ms = original_time - optimized_time
    
    return ValidationTestResponse(
        original_query=request.query,
        original_execution_time_ms=original_result["execution_time_ms"],
        original_row_count=original_result["row_count"],
        original_rows=original_result["rows"],
        optimized_query=optimized_query,
        optimized_execution_time_ms=optimized_result["execution_time_ms"] if optimized_result else None,
        optimized_row_count=optimized_result["row_count"] if optimized_result else None,
        optimized_rows=optimized_result["rows"] if optimized_result else None,
        optimization_rules=optimization_rules,
        recommendations=[
            {
                "type": s.optimization_type.value,
                "description": s.description,
                "hint": s.implementation_hint,
                "estimated_improvement": s.estimated_improvement_pct,
                "confidence": s.confidence
            }
            for s in recommendations.suggestions
        ],
        results_match=results_match,
        improvement_percentage=round(improvement_pct, 2),
        improvement_actual_ms=round(improvement_ms, 3),
        success=True
    )


@router.post("/compare", response_model=CompareQueriesResponse)
async def compare_queries(
    request: CompareQueriesRequest,
    db: AsyncSession = Depends(get_target_db)
):
    """
    Compare two queries side by side.
    
    Execute both queries and compare their results and performance.
    Useful for manually testing different query variations.
    """
    # Execute both queries
    result_a = await execute_query_with_timing(db, request.query_a, request.limit)
    result_b = await execute_query_with_timing(db, request.query_b, request.limit)
    
    if not result_a["success"]:
        return CompareQueriesResponse(
            query_a=request.query_a,
            query_a_time_ms=0,
            query_a_rows=0,
            query_a_results=[],
            query_b=request.query_b,
            query_b_time_ms=0,
            query_b_rows=0,
            query_b_results=[],
            results_match=False,
            improvement_percentage=0,
            faster_query="equal",
            success=False,
            error=f"Failed to execute Query A: {result_a['error']}"
        )
    
    if not result_b["success"]:
        return CompareQueriesResponse(
            query_a=request.query_a,
            query_a_time_ms=result_a["execution_time_ms"],
            query_a_rows=result_a["row_count"],
            query_a_results=result_a["rows"],
            query_b=request.query_b,
            query_b_time_ms=0,
            query_b_rows=0,
            query_b_results=[],
            results_match=False,
            improvement_percentage=0,
            faster_query="equal",
            success=False,
            error=f"Failed to execute Query B: {result_b['error']}"
        )
    
    # Compare results
    results_match = compare_results(result_a["rows"], result_b["rows"])
    
    # Calculate which is faster
    time_a = result_a["execution_time_ms"]
    time_b = result_b["execution_time_ms"]
    
    if abs(time_a - time_b) < 0.1:  # Within 0.1ms is considered equal
        faster_query = "equal"
        improvement_pct = 0
    elif time_a < time_b:
        faster_query = "A"
        improvement_pct = ((time_b - time_a) / time_b) * 100 if time_b > 0 else 0
    else:
        faster_query = "B"
        improvement_pct = ((time_a - time_b) / time_a) * 100 if time_a > 0 else 0
    
    return CompareQueriesResponse(
        query_a=request.query_a,
        query_a_time_ms=result_a["execution_time_ms"],
        query_a_rows=result_a["row_count"],
        query_a_results=result_a["rows"],
        query_b=request.query_b,
        query_b_time_ms=result_b["execution_time_ms"],
        query_b_rows=result_b["row_count"],
        query_b_results=result_b["rows"],
        results_match=results_match,
        improvement_percentage=round(improvement_pct, 2),
        faster_query=faster_query,
        success=True
    )


@router.post("/seed", response_model=SeedDataResponse)
async def seed_test_data(
    request: SeedDataRequest = SeedDataRequest(),
    db: AsyncSession = Depends(get_target_db)
):
    """
    Create test tables and seed with sample data.
    
    Creates the following tables in the target database:
    - categories
    - products
    - customers
    - orders
    - order_items
    
    **WARNING**: If drop_existing is True (default), existing tables will be dropped!
    """
    tables_created = []
    records_inserted = {}
    
    try:
        if request.drop_existing:
            # Drop tables in reverse order (due to foreign keys)
            drop_tables = ["order_items", "orders", "products", "customers", "categories"]
            for table in drop_tables:
                await db.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
            await db.commit()
        
        # Create categories table
        await db.execute(text("""
            CREATE TABLE IF NOT EXISTS categories (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        tables_created.append("categories")
        
        # Create products table
        await db.execute(text("""
            CREATE TABLE IF NOT EXISTS products (
                id SERIAL PRIMARY KEY,
                name VARCHAR(200) NOT NULL,
                description TEXT,
                price DECIMAL(10, 2) NOT NULL,
                stock_quantity INTEGER DEFAULT 0,
                category_id INTEGER REFERENCES categories(id),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        tables_created.append("products")
        
        # Create customers table
        await db.execute(text("""
            CREATE TABLE IF NOT EXISTS customers (
                id SERIAL PRIMARY KEY,
                name VARCHAR(150) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                city VARCHAR(100),
                country VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        tables_created.append("customers")
        
        # Create orders table
        await db.execute(text("""
            CREATE TABLE IF NOT EXISTS orders (
                id SERIAL PRIMARY KEY,
                customer_id INTEGER REFERENCES customers(id),
                status VARCHAR(50) DEFAULT 'pending',
                total_amount DECIMAL(12, 2) DEFAULT 0,
                order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        tables_created.append("orders")
        
        # Create order_items table
        await db.execute(text("""
            CREATE TABLE IF NOT EXISTS order_items (
                id SERIAL PRIMARY KEY,
                order_id INTEGER REFERENCES orders(id),
                product_id INTEGER REFERENCES products(id),
                quantity INTEGER NOT NULL,
                unit_price DECIMAL(10, 2) NOT NULL
            )
        """))
        tables_created.append("order_items")
        
        await db.commit()
        
        # Seed categories
        categories = [
            ("Electronics", "Electronic devices and gadgets"),
            ("Clothing", "Apparel and fashion items"),
            ("Books", "Physical and digital books"),
            ("Home & Garden", "Home improvement and gardening"),
            ("Sports", "Sports equipment and accessories"),
            ("Toys", "Children's toys and games"),
            ("Food & Beverages", "Food items and drinks"),
            ("Health & Beauty", "Health and beauty products"),
            ("Automotive", "Car parts and accessories"),
            ("Office Supplies", "Office and school supplies")
        ]
        for name, desc in categories:
            await db.execute(
                text("INSERT INTO categories (name, description) VALUES (:name, :desc)"),
                {"name": name, "desc": desc}
            )
        records_inserted["categories"] = len(categories)
        
        # Seed products
        product_names = [
            "Wireless Headphones", "Smart Watch", "Laptop Stand", "USB Hub", "Bluetooth Speaker",
            "Cotton T-Shirt", "Denim Jeans", "Running Shoes", "Winter Jacket", "Sunglasses",
            "Mystery Novel", "Cookbook", "Science Fiction", "Biography", "Self-Help Guide",
            "Garden Hose", "Plant Pot", "LED Bulbs", "Tool Set", "Paint Brush",
            "Basketball", "Tennis Racket", "Yoga Mat", "Dumbbells", "Jump Rope"
        ]
        for i in range(request.num_products):
            name = f"{random.choice(product_names)} {i+1}"
            price = round(random.uniform(9.99, 499.99), 2)
            stock = random.randint(0, 500)
            category_id = random.randint(1, 10)
            is_active = random.random() > 0.1
            await db.execute(
                text("""
                    INSERT INTO products (name, description, price, stock_quantity, category_id, is_active)
                    VALUES (:name, :desc, :price, :stock, :cat_id, :active)
                """),
                {
                    "name": name,
                    "desc": f"Description for {name}",
                    "price": price,
                    "stock": stock,
                    "cat_id": category_id,
                    "active": is_active
                }
            )
        records_inserted["products"] = request.num_products
        
        # Seed customers
        first_names = ["John", "Jane", "Mike", "Sarah", "David", "Emily", "Chris", "Lisa", "Tom", "Anna"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Wilson", "Moore"]
        cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin"]
        countries = ["USA", "Canada", "UK", "Germany", "France", "Australia", "Japan", "Brazil", "Mexico", "India"]
        
        for i in range(request.num_customers):
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
            email = f"customer{i+1}@example.com"
            city = random.choice(cities)
            country = random.choice(countries)
            await db.execute(
                text("""
                    INSERT INTO customers (name, email, city, country)
                    VALUES (:name, :email, :city, :country)
                """),
                {"name": name, "email": email, "city": city, "country": country}
            )
        records_inserted["customers"] = request.num_customers
        
        # Seed orders
        statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
        order_items_count = 0
        
        for i in range(request.num_orders):
            customer_id = random.randint(1, request.num_customers)
            status = random.choice(statuses)
            order_date = datetime.now() - timedelta(days=random.randint(0, 365))
            
            await db.execute(
                text("""
                    INSERT INTO orders (customer_id, status, total_amount, order_date)
                    VALUES (:cust_id, :status, 0, :order_date)
                """),
                {"cust_id": customer_id, "status": status, "order_date": order_date}
            )
            
            # Add order items (1-5 items per order)
            num_items = random.randint(1, 5)
            order_id = i + 1
            total = 0
            
            for _ in range(num_items):
                product_id = random.randint(1, request.num_products)
                quantity = random.randint(1, 5)
                unit_price = round(random.uniform(9.99, 199.99), 2)
                total += quantity * unit_price
                
                await db.execute(
                    text("""
                        INSERT INTO order_items (order_id, product_id, quantity, unit_price)
                        VALUES (:order_id, :prod_id, :qty, :price)
                    """),
                    {"order_id": order_id, "prod_id": product_id, "qty": quantity, "price": unit_price}
                )
                order_items_count += 1
            
            # Update order total
            await db.execute(
                text("UPDATE orders SET total_amount = :total WHERE id = :id"),
                {"total": round(total, 2), "id": order_id}
            )
        
        records_inserted["orders"] = request.num_orders
        records_inserted["order_items"] = order_items_count
        
        await db.commit()
        
        return SeedDataResponse(
            success=True,
            tables_created=tables_created,
            records_inserted=records_inserted,
            message=f"Successfully created {len(tables_created)} tables and seeded with test data"
        )
        
    except Exception as e:
        await db.rollback()
        return SeedDataResponse(
            success=False,
            tables_created=tables_created,
            records_inserted=records_inserted,
            message="Failed to seed data",
            error=str(e)
        )


@router.get("/tables", response_model=ListTablesResponse)
async def list_test_tables(
    db: AsyncSession = Depends(get_target_db)
):
    """
    List available test tables in the target database.
    
    Returns table names, row counts, and column information.
    """
    test_tables = ["categories", "products", "customers", "orders", "order_items"]
    tables_info = []
    
    try:
        for table_name in test_tables:
            # Check if table exists and get row count
            try:
                count_result = await db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                row_count = count_result.scalar() or 0
                
                # Get column names
                columns_result = await db.execute(text(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = :table_name
                    ORDER BY ordinal_position
                """), {"table_name": table_name})
                columns = [row[0] for row in columns_result.fetchall()]
                
                tables_info.append(TableInfo(
                    name=table_name,
                    row_count=row_count,
                    columns=columns
                ))
            except Exception:
                # Table doesn't exist
                pass
        
        return ListTablesResponse(
            tables=tables_info,
            success=True
        )
        
    except Exception as e:
        return ListTablesResponse(
            tables=[],
            success=False,
            error=str(e)
        )


@router.get("/sample-queries")
async def get_sample_queries():
    """
    Get sample queries to test with the validation endpoints.
    
    Returns a list of example queries that work with the seeded test data.
    """
    return {
        "simple_queries": [
            {
                "description": "Select all products (will trigger SELECT * optimization)",
                "query": "SELECT * FROM products"
            },
            {
                "description": "Products by category",
                "query": "SELECT * FROM products WHERE category_id = 5"
            },
            {
                "description": "Expensive products",
                "query": "SELECT * FROM products WHERE price > 100 ORDER BY price DESC"
            }
        ],
        "join_queries": [
            {
                "description": "Products with category names",
                "query": "SELECT p.name, p.price, c.name as category FROM products p JOIN categories c ON p.category_id = c.id"
            },
            {
                "description": "Orders with customer info",
                "query": "SELECT o.id, o.total_amount, c.name, c.email FROM orders o JOIN customers c ON o.customer_id = c.id WHERE o.status = 'pending'"
            },
            {
                "description": "Order details with products",
                "query": "SELECT o.id, p.name, oi.quantity, oi.unit_price FROM orders o JOIN order_items oi ON o.id = oi.order_id JOIN products p ON oi.product_id = p.id LIMIT 50"
            }
        ],
        "optimization_candidates": [
            {
                "description": "OR conditions (can be optimized to IN)",
                "query": "SELECT * FROM products WHERE category_id = 1 OR category_id = 2 OR category_id = 3"
            },
            {
                "description": "SELECT * without LIMIT (will add LIMIT)",
                "query": "SELECT * FROM orders WHERE status = 'delivered'"
            },
            {
                "description": "Complex query with multiple conditions",
                "query": "SELECT * FROM products WHERE category_id = 5 AND price > 50 AND is_active = true ORDER BY price DESC"
            }
        ]
    }
