#!/usr/bin/env python3
"""
NeuroQO API Test Script

This script tests all the main API endpoints to verify the system is working.
Run this after starting the server with: uvicorn main:app --reload
"""
import httpx
import asyncio
import sys

BASE_URL = "http://localhost:8000/api/v1"
token = None


async def test_health():
    """Test if the server is running."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/docs")
            return response.status_code == 200
        except httpx.ConnectError:
            return False


async def test_auth():
    """Test authentication endpoints."""
    global token
    async with httpx.AsyncClient() as client:
        print("\n📌 Testing Authentication...")
        
        # Register
        print("   → Registering new user...")
        response = await client.post(f"{BASE_URL}/auth/register", json={
            "email": "testuser@neuroqo.com",
            "username": "testuser",
            "password": "testpass123",
            "full_name": "Test User"
        })
        if response.status_code == 201:
            print("   ✅ User registered successfully")
        elif response.status_code == 400 and "already" in response.text.lower():
            print("   ℹ️  User already exists (OK)")
        else:
            print(f"   ❌ Registration failed: {response.status_code} - {response.text}")
            return False
        
        # Login
        print("   → Logging in...")
        response = await client.post(f"{BASE_URL}/auth/token", data={
            "username": "testuser",
            "password": "testpass123"
        })
        if response.status_code == 200:
            token = response.json()["access_token"]
            print(f"   ✅ Login successful, token: {token[:30]}...")
            return True
        else:
            print(f"   ❌ Login failed: {response.status_code} - {response.text}")
            return False


async def test_queries():
    """Test query endpoints."""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}
        print("\n📌 Testing Query Endpoints...")
        
        # Analyze query
        print("   → Analyzing a query...")
        response = await client.post(f"{BASE_URL}/queries/analyze", 
            headers=headers,
            json={"query_text": "SELECT * FROM users WHERE email = 'test@example.com' AND status = 'active' ORDER BY created_at DESC"}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Query analyzed: type={data.get('query_type')}, complexity={data.get('complexity')}")
        else:
            print(f"   ❌ Analysis failed: {response.status_code}")
        
        # Log query
        print("   → Logging a query...")
        response = await client.post(f"{BASE_URL}/queries/log",
            headers=headers,
            json={
                "query_text": "SELECT id, name, price FROM products WHERE category_id = 5 AND price > 50",
                "execution_time_ms": 350,
                "database_name": "test_db"
            }
        )
        if response.status_code in [200, 201]:
            print("   ✅ Query logged successfully")
        else:
            print(f"   ❌ Logging failed: {response.status_code} - {response.text}")
        
        # List queries
        print("   → Listing queries...")
        response = await client.get(f"{BASE_URL}/queries?page=1&page_size=5", headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Found {data.get('total', 0)} queries")
        else:
            print(f"   ❌ List failed: {response.status_code}")
        
        # Get slow queries
        print("   → Getting slow queries...")
        response = await client.get(f"{BASE_URL}/queries/slow?threshold_ms=100", headers=headers)
        if response.status_code == 200:
            data = response.json()
            count = len(data) if isinstance(data, list) else data.get('total', 0)
            print(f"   ✅ Found {count} slow queries")
        else:
            print(f"   ❌ Slow queries failed: {response.status_code}")


async def test_optimizations():
    """Test optimization endpoints."""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}
        print("\n📌 Testing Optimization Endpoints...")
        
        # Generate optimization
        print("   → Generating optimization...")
        response = await client.post(f"{BASE_URL}/optimizations/generate",
            headers=headers,
            json={"query_text": "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id WHERE o.status = 'pending'"}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Optimization generated: {data.get('optimization_type', 'N/A')}")
        else:
            print(f"   ⚠️  Generation returned: {response.status_code}")
        
        # List optimizations
        print("   → Listing optimizations...")
        response = await client.get(f"{BASE_URL}/optimizations?page=1", headers=headers)
        if response.status_code == 200:
            data = response.json()
            total = data.get('total', 0) if isinstance(data, dict) else len(data)
            print(f"   ✅ Found {total} optimizations")
        else:
            print(f"   ❌ List failed: {response.status_code}")


async def test_indexes():
    """Test index advisor endpoints."""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}
        print("\n📌 Testing Index Advisor Endpoints...")
        
        # Analyze for indexes
        print("   → Analyzing query for index recommendations...")
        response = await client.post(f"{BASE_URL}/indexes/analyze",
            headers=headers,
            json={"query_text": "SELECT * FROM orders WHERE customer_id = 123 AND order_date > '2024-01-01' ORDER BY total DESC"}
        )
        if response.status_code == 200:
            data = response.json()
            recs = data.get('recommendations', [])
            print(f"   ✅ Got {len(recs)} index recommendations")
        else:
            print(f"   ⚠️  Analysis returned: {response.status_code}")
        
        # List recommendations
        print("   → Listing index recommendations...")
        response = await client.get(f"{BASE_URL}/indexes/recommendations", headers=headers)
        if response.status_code == 200:
            data = response.json()
            total = data.get('total', 0) if isinstance(data, dict) else len(data)
            print(f"   ✅ Found {total} recommendations")
        else:
            print(f"   ❌ List failed: {response.status_code}")


async def test_experiments():
    """Test A/B experiment endpoints."""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}
        print("\n📌 Testing Experiment Endpoints...")
        
        # Create experiment
        print("   → Creating experiment...")
        response = await client.post(f"{BASE_URL}/experiments",
            headers=headers,
            json={
                "name": "Test Index Experiment",
                "description": "Testing performance improvement",
                "control_query": "SELECT * FROM products WHERE category = 'electronics'",
                "treatment_query": "SELECT id, name, price FROM products WHERE category = 'electronics'",
                "success_metric": "execution_time",
                "sample_size": 100
            }
        )
        if response.status_code in [200, 201]:
            data = response.json()
            exp_id = data.get('id', 1)
            print(f"   ✅ Experiment created with ID: {exp_id}")
        elif response.status_code == 400 and "exists" in response.text.lower():
            print("   ℹ️  Experiment already exists (OK)")
            exp_id = 1
        else:
            print(f"   ⚠️  Creation returned: {response.status_code}")
            exp_id = None
        
        # List experiments
        print("   → Listing experiments...")
        response = await client.get(f"{BASE_URL}/experiments", headers=headers)
        if response.status_code == 200:
            data = response.json()
            total = data.get('total', len(data)) if isinstance(data, dict) else len(data)
            print(f"   ✅ Found {total} experiments")
        else:
            print(f"   ❌ List failed: {response.status_code}")


async def test_dashboard():
    """Test dashboard endpoints."""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}
        print("\n📌 Testing Dashboard Endpoints...")
        
        # Overview
        print("   → Getting overview...")
        response = await client.get(f"{BASE_URL}/dashboard/overview", headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Overview: {data.get('total_queries', 0)} queries, {data.get('total_optimizations', 0)} optimizations")
        else:
            print(f"   ❌ Overview failed: {response.status_code}")
        
        # Trends
        print("   → Getting trends...")
        response = await client.get(f"{BASE_URL}/dashboard/trends?days=7", headers=headers)
        if response.status_code == 200:
            print("   ✅ Trends retrieved")
        else:
            print(f"   ⚠️  Trends returned: {response.status_code}")
        
        # Alerts
        print("   → Getting alerts...")
        response = await client.get(f"{BASE_URL}/dashboard/alerts", headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Found {data.get('count', 0)} alerts")
        else:
            print(f"   ⚠️  Alerts returned: {response.status_code}")


async def test_models():
    """Test ML model endpoints."""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}
        print("\n📌 Testing ML Model Endpoints...")
        
        # Model status
        print("   → Getting model statuses...")
        response = await client.get(f"{BASE_URL}/models/status", headers=headers)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict):
                models = data.get('models', data)
                print(f"   ✅ Found {len(models)} models")
            else:
                print(f"   ✅ Model status retrieved")
        else:
            print(f"   ⚠️  Status returned: {response.status_code}")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("🚀 NeuroQO API Test Suite")
    print("=" * 60)
    
    # Check server health
    print("\n🔍 Checking server health...")
    if not await test_health():
        print("❌ Server is not running!")
        print("\n💡 Start the server with: uvicorn main:app --reload")
        sys.exit(1)
    print("✅ Server is running!")
    
    # Run tests
    if not await test_auth():
        print("\n❌ Authentication failed, cannot continue")
        sys.exit(1)
    
    await test_queries()
    await test_optimizations()
    await test_indexes()
    await test_experiments()
    await test_dashboard()
    await test_models()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    print("\n📖 Open http://localhost:8000/docs for interactive API documentation")


if __name__ == "__main__":
    asyncio.run(main())
