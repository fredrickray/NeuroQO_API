# NeuroQO - AI-Driven Adaptive Query Optimizer

An intelligent system that observes, analyzes, and optimizes database queries using machine learning techniques.

## Features

- **Query Explorer** - Browse and analyze captured queries
- **Slow Query Profiler** - Detect and profile slow-running queries
- **ML-Based Optimization** - Automatic query rewriting and optimization suggestions
- **Index Advisor** - Intelligent index recommendations
- **A/B Experiments** - Test optimization strategies with statistical analysis
- **Model Monitoring** - Track ML model performance and detect drift

## Prerequisites

- Python 3.10+
- PostgreSQL 14+ (primary database)
- Optional: MySQL 8+ (if optimizing MySQL databases)

## Quick Start

### 1. Set Up Virtual Environment

```bash
cd /Users/fredrickanyanwu/Documents/NeuroQO
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the project root:

```env
# Application
APP_NAME=NeuroQO
DEBUG=true

# Main Database (where NeuroQO stores its data)
DATABASE_TYPE=postgresql
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_USER=postgres
DATABASE_PASSWORD=your_password
DATABASE_NAME=neuroqo

# Target Database (the database you want to optimize)
TARGET_DB_HOST=localhost
TARGET_DB_PORT=5432
TARGET_DB_USER=postgres
TARGET_DB_PASSWORD=your_password
TARGET_DB_NAME=your_target_db

# Security
SECRET_KEY=your-super-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# ML Settings
MODEL_PATH=models/
SLOW_QUERY_THRESHOLD_MS=1000
```

### 4. Create Databases

```bash
# Connect to PostgreSQL
psql -U postgres

# Create the NeuroQO database
CREATE DATABASE neuroqo;

# Optionally create a target test database
CREATE DATABASE target_db;

\q
```

### 5. Run Database Migrations

```bash
alembic upgrade head
```

### 6. Start the Application

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

- **API Docs (Swagger)**: http://localhost:8000/docs
- **API Docs (ReDoc)**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Testing All Functionalities

### Using the Interactive API Docs

Open http://localhost:8000/docs in your browser for an interactive Swagger UI.

### API Endpoints Overview

#### 1. Authentication (`/api/v1/auth`)

```bash
# Register a new user
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@example.com",
    "username": "admin",
    "password": "password123",
    "full_name": "Admin User"
  }'

# Login and get token
curl -X POST "http://localhost:8000/api/v1/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=password123"

# Save the access_token from the response for subsequent requests
export TOKEN="your_access_token_here"

# Get current user info
curl -X GET "http://localhost:8000/api/v1/auth/me" \
  -H "Authorization: Bearer $TOKEN"
```

#### 2. Query Management (`/api/v1/queries`)

```bash
# Log a new query for analysis
curl -X POST "http://localhost:8000/api/v1/queries/log" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "SELECT * FROM users WHERE email = '\''test@example.com'\'' ORDER BY created_at",
    "execution_time_ms": 1500,
    "database_name": "target_db"
  }'

# Analyze a query
curl -X POST "http://localhost:8000/api/v1/queries/analyze" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id WHERE o.status = '\''pending'\''"
  }'

# List all queries
curl -X GET "http://localhost:8000/api/v1/queries?page=1&page_size=10" \
  -H "Authorization: Bearer $TOKEN"

# Get slow queries
curl -X GET "http://localhost:8000/api/v1/queries/slow?threshold_ms=500" \
  -H "Authorization: Bearer $TOKEN"

# Get query patterns
curl -X GET "http://localhost:8000/api/v1/queries/patterns" \
  -H "Authorization: Bearer $TOKEN"
```

#### 3. Optimizations (`/api/v1/optimizations`)

```bash
# Generate optimizations for a query
curl -X POST "http://localhost:8000/api/v1/optimizations/generate" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "SELECT * FROM products WHERE category_id = 5 AND price > 100 ORDER BY name"
  }'

# List all optimizations
curl -X GET "http://localhost:8000/api/v1/optimizations?page=1" \
  -H "Authorization: Bearer $TOKEN"

# Apply an optimization
curl -X POST "http://localhost:8000/api/v1/optimizations/1/apply" \
  -H "Authorization: Bearer $TOKEN"

# Rollback an optimization
curl -X POST "http://localhost:8000/api/v1/optimizations/1/rollback" \
  -H "Authorization: Bearer $TOKEN"
```

#### 4. Index Advisor (`/api/v1/indexes`)

```bash
# Analyze a query for index recommendations
curl -X POST "http://localhost:8000/api/v1/indexes/analyze" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "SELECT * FROM orders WHERE customer_id = 123 AND order_date > '\''2024-01-01'\''"
  }'

# List index recommendations
curl -X GET "http://localhost:8000/api/v1/indexes/recommendations" \
  -H "Authorization: Bearer $TOKEN"

# Apply an index recommendation
curl -X POST "http://localhost:8000/api/v1/indexes/action" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "recommendation_id": 1,
    "action": "apply"
  }'

# Get existing indexes
curl -X GET "http://localhost:8000/api/v1/indexes/existing" \
  -H "Authorization: Bearer $TOKEN"

# Get index history
curl -X GET "http://localhost:8000/api/v1/indexes/history" \
  -H "Authorization: Bearer $TOKEN"
```

#### 5. A/B Experiments (`/api/v1/experiments`)

```bash
# Create an experiment
curl -X POST "http://localhost:8000/api/v1/experiments" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Index Performance Test",
    "description": "Testing new index on orders table",
    "control_query": "SELECT * FROM orders WHERE status = '\''pending'\''",
    "treatment_query": "SELECT * FROM orders USE INDEX (idx_status) WHERE status = '\''pending'\''",
    "success_metric": "execution_time",
    "sample_size": 1000
  }'

# List experiments
curl -X GET "http://localhost:8000/api/v1/experiments" \
  -H "Authorization: Bearer $TOKEN"

# Start an experiment
curl -X POST "http://localhost:8000/api/v1/experiments/1/start" \
  -H "Authorization: Bearer $TOKEN"

# Record experiment result
curl -X POST "http://localhost:8000/api/v1/experiments/1/results" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "variant": "control",
    "execution_time_ms": 150,
    "success": true
  }'

# Get experiment results/analysis
curl -X GET "http://localhost:8000/api/v1/experiments/1/results" \
  -H "Authorization: Bearer $TOKEN"

# Stop an experiment
curl -X POST "http://localhost:8000/api/v1/experiments/1/stop" \
  -H "Authorization: Bearer $TOKEN"
```

#### 6. Dashboard & Monitoring (`/api/v1/dashboard`)

```bash
# Get overview statistics
curl -X GET "http://localhost:8000/api/v1/dashboard/overview" \
  -H "Authorization: Bearer $TOKEN"

# Get query trends
curl -X GET "http://localhost:8000/api/v1/dashboard/trends?days=7" \
  -H "Authorization: Bearer $TOKEN"

# Get top query patterns
curl -X GET "http://localhost:8000/api/v1/dashboard/patterns?limit=10" \
  -H "Authorization: Bearer $TOKEN"

# Get optimization impact
curl -X GET "http://localhost:8000/api/v1/dashboard/optimizations/impact?days=30" \
  -H "Authorization: Bearer $TOKEN"

# Get ML model status
curl -X GET "http://localhost:8000/api/v1/dashboard/models/status" \
  -H "Authorization: Bearer $TOKEN"

# Get system alerts
curl -X GET "http://localhost:8000/api/v1/dashboard/alerts" \
  -H "Authorization: Bearer $TOKEN"

# Get before/after comparison
curl -X GET "http://localhost:8000/api/v1/dashboard/comparison?days=30" \
  -H "Authorization: Bearer $TOKEN"
```

#### 7. ML Models (`/api/v1/models`)

```bash
# Get all model statuses
curl -X GET "http://localhost:8000/api/v1/models/status" \
  -H "Authorization: Bearer $TOKEN"

# Get specific model status
curl -X GET "http://localhost:8000/api/v1/models/query_classifier/status" \
  -H "Authorization: Bearer $TOKEN"

# Train a model (requires training data)
curl -X POST "http://localhost:8000/api/v1/models/query_classifier/train" \
  -H "Authorization: Bearer $TOKEN"

# Check for model drift
curl -X GET "http://localhost:8000/api/v1/models/query_classifier/drift" \
  -H "Authorization: Bearer $TOKEN"

# Get model metrics
curl -X GET "http://localhost:8000/api/v1/models/query_classifier/metrics" \
  -H "Authorization: Bearer $TOKEN"

# Export a model
curl -X GET "http://localhost:8000/api/v1/models/query_classifier/export" \
  -H "Authorization: Bearer $TOKEN" \
  --output model_export.json

# Get training history
curl -X GET "http://localhost:8000/api/v1/models/history" \
  -H "Authorization: Bearer $TOKEN"
```

## Testing with Python

Create a test script `test_api.py`:

```python
import httpx
import asyncio

BASE_URL = "http://localhost:8000/api/v1"

async def main():
    async with httpx.AsyncClient() as client:
        # 1. Register user
        print("1. Registering user...")
        response = await client.post(f"{BASE_URL}/auth/register", json={
            "email": "test@example.com",
            "username": "testuser",
            "password": "testpass123",
            "full_name": "Test User"
        })
        print(f"   Status: {response.status_code}")

        # 2. Login
        print("2. Logging in...")
        response = await client.post(f"{BASE_URL}/auth/token", data={
            "username": "testuser",
            "password": "testpass123"
        })
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print(f"   Got token: {token[:20]}...")

        # 3. Analyze a query
        print("3. Analyzing query...")
        response = await client.post(f"{BASE_URL}/queries/analyze",
            headers=headers,
            json={"query_text": "SELECT * FROM users WHERE status = 'active' ORDER BY created_at DESC"}
        )
        print(f"   Analysis: {response.json()}")

        # 4. Log a query
        print("4. Logging query...")
        response = await client.post(f"{BASE_URL}/queries/log",
            headers=headers,
            json={
                "query_text": "SELECT * FROM products WHERE price > 100",
                "execution_time_ms": 250,
                "database_name": "test_db"
            }
        )
        print(f"   Status: {response.status_code}")

        # 5. Generate optimization
        print("5. Generating optimization...")
        response = await client.post(f"{BASE_URL}/optimizations/generate",
            headers=headers,
            json={"query_text": "SELECT * FROM orders WHERE customer_id = 1"}
        )
        print(f"   Optimization: {response.json()}")

        # 6. Get dashboard overview
        print("6. Getting dashboard overview...")
        response = await client.get(f"{BASE_URL}/dashboard/overview", headers=headers)
        print(f"   Overview: {response.json()}")

        print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python test_api.py
```

## Running Automated Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_queries.py -v
```

## Project Structure

```
NeuroQO/
├── main.py              # FastAPI application entry point
├── requirements.txt     # Python dependencies
├── alembic.ini         # Database migration config
├── alembic/            # Migration scripts
├── app/
│   ├── api/routes/     # API endpoints
│   │   ├── auth.py     # Authentication
│   │   ├── queries.py  # Query management
│   │   ├── optimizations.py
│   │   ├── indexes.py  # Index advisor
│   │   ├── experiments.py  # A/B testing
│   │   ├── dashboard.py    # Monitoring
│   │   └── models.py   # ML model management
│   ├── core/           # Core utilities
│   │   ├── config.py   # Settings
│   │   ├── database.py # DB connection
│   │   └── security.py # Auth utilities
│   ├── ml/             # Machine learning
│   │   ├── query_classifier.py
│   │   ├── performance_predictor.py
│   │   ├── optimization_recommender.py
│   │   ├── feature_extractor.py
│   │   └── model_manager.py
│   ├── models/         # SQLAlchemy models
│   ├── schemas/        # Pydantic schemas
│   └── services/       # Business logic
│       ├── query_analyzer.py
│       ├── query_profiler.py
│       ├── query_rewriter.py
│       ├── index_advisor.py
│       └── cache_manager.py
```

## Troubleshooting

### Database Connection Issues

- Ensure PostgreSQL is running: `brew services start postgresql` (macOS)
- Verify credentials in `.env`
- Check database exists: `psql -U postgres -c "\l"`

### Import Errors

- Activate virtual environment: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

### Migration Errors

- Reset migrations: `alembic downgrade base && alembic upgrade head`

## License

MIT License
