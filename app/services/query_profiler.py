"""
Query Profiler Service - Captures and profiles query execution from database.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.query_analyzer import QueryAnalyzerService


@dataclass
class QueryExecutionProfile:
    """Profile of a query execution."""
    query_text: str
    execution_time_ms: float
    rows_examined: int
    rows_returned: int
    execution_plan: Dict[str, Any]
    buffer_reads: Optional[int] = None
    buffer_hits: Optional[int] = None
    temp_written: Optional[int] = None


@dataclass
class SlowLogEntry:
    """Entry from the slow query log."""
    query_text: str
    execution_time_ms: float
    lock_time_ms: float
    rows_examined: int
    rows_sent: int
    timestamp: datetime
    user: str
    database: str


class QueryProfilerService:
    """Service for profiling database queries."""
    
    def __init__(self, db_type: str = "postgresql"):
        self.db_type = db_type
        self.analyzer = QueryAnalyzerService()
        self._slow_log_cache: List[SlowLogEntry] = []
    
    async def explain_query(
        self, 
        session: AsyncSession, 
        query: str,
        analyze: bool = False
    ) -> Dict[str, Any]:
        """
        Get the execution plan for a query.
        
        Args:
            session: Database session
            query: SQL query to explain
            analyze: If True, actually execute the query for real statistics
            
        Returns:
            Execution plan as a dictionary
        """
        if self.db_type == "postgresql":
            return await self._explain_postgresql(session, query, analyze)
        else:
            return await self._explain_mysql(session, query, analyze)
    
    async def _explain_postgresql(
        self, 
        session: AsyncSession, 
        query: str,
        analyze: bool
    ) -> Dict[str, Any]:
        """Get PostgreSQL execution plan."""
        explain_query = f"EXPLAIN (FORMAT JSON, COSTS, VERBOSE, BUFFERS"
        if analyze:
            explain_query += ", ANALYZE"
        explain_query += f") {query}"
        
        try:
            result = await session.execute(text(explain_query))
            plan = result.fetchone()
            if plan:
                return {
                    "plan": plan[0],
                    "db_type": "postgresql"
                }
        except Exception as e:
            return {
                "error": str(e),
                "db_type": "postgresql"
            }
        
        return {"plan": None, "db_type": "postgresql"}
    
    async def _explain_mysql(
        self, 
        session: AsyncSession, 
        query: str,
        analyze: bool
    ) -> Dict[str, Any]:
        """Get MySQL execution plan."""
        explain_query = f"EXPLAIN FORMAT=JSON {query}"
        
        try:
            result = await session.execute(text(explain_query))
            plan = result.fetchone()
            if plan:
                return {
                    "plan": plan[0],
                    "db_type": "mysql"
                }
        except Exception as e:
            return {
                "error": str(e),
                "db_type": "mysql"
            }
        
        return {"plan": None, "db_type": "mysql"}
    
    async def profile_query(
        self, 
        session: AsyncSession, 
        query: str
    ) -> QueryExecutionProfile:
        """
        Profile a query execution with detailed metrics.
        
        Args:
            session: Database session
            query: SQL query to profile
            
        Returns:
            QueryExecutionProfile with execution metrics
        """
        # Get execution plan with analysis
        plan = await self.explain_query(session, query, analyze=True)
        
        # Extract metrics from plan
        metrics = self._extract_metrics_from_plan(plan)
        
        return QueryExecutionProfile(
            query_text=query,
            execution_time_ms=metrics.get("execution_time_ms", 0),
            rows_examined=metrics.get("rows_examined", 0),
            rows_returned=metrics.get("rows_returned", 0),
            execution_plan=plan,
            buffer_reads=metrics.get("buffer_reads"),
            buffer_hits=metrics.get("buffer_hits"),
            temp_written=metrics.get("temp_written")
        )
    
    def _extract_metrics_from_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Extract execution metrics from the plan."""
        metrics = {
            "execution_time_ms": 0,
            "rows_examined": 0,
            "rows_returned": 0,
            "buffer_reads": 0,
            "buffer_hits": 0,
            "temp_written": 0
        }
        
        if not plan or "error" in plan:
            return metrics
        
        plan_data = plan.get("plan", [])
        
        if plan.get("db_type") == "postgresql" and isinstance(plan_data, list) and plan_data:
            pg_plan = plan_data[0].get("Plan", {})
            metrics["execution_time_ms"] = plan_data[0].get("Execution Time", 0)
            metrics["rows_returned"] = pg_plan.get("Actual Rows", 0)
            metrics["rows_examined"] = pg_plan.get("Actual Total Rows", 
                                                    pg_plan.get("Actual Rows", 0))
            
            # Buffer statistics
            shared_hit = pg_plan.get("Shared Hit Blocks", 0)
            shared_read = pg_plan.get("Shared Read Blocks", 0)
            metrics["buffer_hits"] = shared_hit
            metrics["buffer_reads"] = shared_read
            metrics["temp_written"] = pg_plan.get("Temp Written Blocks", 0)
        
        elif plan.get("db_type") == "mysql" and isinstance(plan_data, dict):
            query_block = plan_data.get("query_block", {})
            cost_info = query_block.get("cost_info", {})
            metrics["execution_time_ms"] = float(cost_info.get("query_cost", 0))  # type: ignore[assignment]
            
            # Try to get rows examined
            table_info = query_block.get("table", {})
            metrics["rows_examined"] = table_info.get("rows_examined_per_scan", 0)
            metrics["rows_returned"] = table_info.get("rows_produced_per_join", 0)
        
        return metrics
    
    async def get_slow_queries(
        self, 
        session: AsyncSession,
        threshold_ms: Optional[float] = None,
        limit: int = 100,
        since: Optional[datetime] = None
    ) -> List[SlowLogEntry]:
        """
        Get slow queries from the database slow log.
        
        Args:
            session: Database session
            threshold_ms: Minimum execution time to consider slow
            limit: Maximum number of queries to return
            since: Only get queries after this timestamp
            
        Returns:
            List of slow log entries
        """
        if threshold_ms is None:
            threshold_ms = settings.SLOW_QUERY_THRESHOLD_MS
        
        if since is None:
            since = datetime.utcnow() - timedelta(days=1)
        
        if self.db_type == "postgresql":
            return await self._get_slow_queries_postgresql(
                session, threshold_ms, limit, since
            )
        else:
            return await self._get_slow_queries_mysql(
                session, threshold_ms, limit, since
            )
    
    async def _get_slow_queries_postgresql(
        self,
        session: AsyncSession,
        threshold_ms: float,
        limit: int,
        since: datetime
    ) -> List[SlowLogEntry]:
        """Get slow queries from PostgreSQL pg_stat_statements."""
        query = text("""
            SELECT 
                query,
                mean_exec_time as avg_time_ms,
                total_exec_time as total_time_ms,
                calls,
                rows,
                shared_blks_hit,
                shared_blks_read
            FROM pg_stat_statements
            WHERE mean_exec_time > :threshold
            ORDER BY mean_exec_time DESC
            LIMIT :limit
        """)
        
        try:
            result = await session.execute(
                query, 
                {"threshold": threshold_ms, "limit": limit}
            )
            rows = result.fetchall()
            
            entries = []
            for row in rows:
                entries.append(SlowLogEntry(
                    query_text=row[0],
                    execution_time_ms=row[1],
                    lock_time_ms=0,
                    rows_examined=row[5] + row[6],  # hits + reads
                    rows_sent=row[4],
                    timestamp=datetime.utcnow(),
                    user="",
                    database=""
                ))
            return entries
        except Exception as e:
            # pg_stat_statements might not be enabled
            return []
    
    async def _get_slow_queries_mysql(
        self,
        session: AsyncSession,
        threshold_ms: float,
        limit: int,
        since: datetime
    ) -> List[SlowLogEntry]:
        """Get slow queries from MySQL slow log or performance schema."""
        query = text("""
            SELECT 
                DIGEST_TEXT as query_text,
                AVG_TIMER_WAIT/1000000000 as avg_time_ms,
                SUM_TIMER_WAIT/1000000000 as total_time_ms,
                COUNT_STAR as calls,
                SUM_ROWS_EXAMINED as rows_examined,
                SUM_ROWS_SENT as rows_sent,
                FIRST_SEEN,
                LAST_SEEN
            FROM performance_schema.events_statements_summary_by_digest
            WHERE AVG_TIMER_WAIT/1000000000 > :threshold
            AND LAST_SEEN > :since
            ORDER BY AVG_TIMER_WAIT DESC
            LIMIT :limit
        """)
        
        try:
            result = await session.execute(
                query,
                {"threshold": threshold_ms, "since": since, "limit": limit}
            )
            rows = result.fetchall()
            
            entries = []
            for row in rows:
                entries.append(SlowLogEntry(
                    query_text=row[0],
                    execution_time_ms=row[1],
                    lock_time_ms=0,
                    rows_examined=row[4],
                    rows_sent=row[5],
                    timestamp=row[7],
                    user="",
                    database=""
                ))
            return entries
        except Exception as e:
            return []
    
    async def get_table_statistics(
        self, 
        session: AsyncSession, 
        table_name: str
    ) -> Dict[str, Any]:
        """Get statistics for a specific table."""
        if self.db_type == "postgresql":
            query = text("""
                SELECT 
                    relname as table_name,
                    n_live_tup as row_count,
                    n_dead_tup as dead_rows,
                    last_vacuum,
                    last_analyze,
                    seq_scan,
                    seq_tup_read,
                    idx_scan,
                    idx_tup_fetch
                FROM pg_stat_user_tables
                WHERE relname = :table_name
            """)
        else:
            query = text("""
                SELECT 
                    TABLE_NAME as table_name,
                    TABLE_ROWS as row_count,
                    DATA_LENGTH as data_size,
                    INDEX_LENGTH as index_size,
                    UPDATE_TIME as last_update
                FROM information_schema.TABLES
                WHERE TABLE_NAME = :table_name
            """)
        
        try:
            result = await session.execute(query, {"table_name": table_name})
            row = result.fetchone()
            if row:
                return dict(row._mapping)
        except Exception as e:
            pass
        
        return {}
    
    async def get_index_usage(
        self, 
        session: AsyncSession, 
        table_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get index usage statistics."""
        if self.db_type == "postgresql":
            query = text("""
                SELECT 
                    schemaname,
                    relname as table_name,
                    indexrelname as index_name,
                    idx_scan as scans,
                    idx_tup_read as tuples_read,
                    idx_tup_fetch as tuples_fetched
                FROM pg_stat_user_indexes
                WHERE (:table_name IS NULL OR relname = :table_name)
                ORDER BY idx_scan DESC
            """)
        else:
            query = text("""
                SELECT 
                    TABLE_NAME as table_name,
                    INDEX_NAME as index_name,
                    CARDINALITY as cardinality,
                    INDEX_TYPE as index_type
                FROM information_schema.STATISTICS
                WHERE (:table_name IS NULL OR TABLE_NAME = :table_name)
            """)
        
        try:
            result = await session.execute(query, {"table_name": table_name})
            rows = result.fetchall()
            return [dict(row._mapping) for row in rows]
        except Exception as e:
            return []
    
    def classify_query_performance(
        self, 
        execution_time_ms: float
    ) -> str:
        """Classify query performance based on execution time."""
        if execution_time_ms < 10:
            return "excellent"
        elif execution_time_ms < 100:
            return "good"
        elif execution_time_ms < 500:
            return "acceptable"
        elif execution_time_ms < settings.SLOW_QUERY_THRESHOLD_MS:
            return "slow"
        else:
            return "critical"
