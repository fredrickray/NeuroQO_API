"""
Index Advisor Service - ML-based index recommendation engine.
"""
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.query_analyzer import QueryAnalyzerService, QueryAnalysis


@dataclass
class IndexRecommendation:
    """Recommendation for a new index."""
    table_name: str
    column_names: List[str]
    index_type: str
    index_name: str
    create_statement: str
    drop_statement: str
    estimated_improvement_ms: float
    estimated_storage_mb: float
    confidence: float
    reasoning: str
    affected_queries: List[str]
    priority: int


@dataclass
class IndexAnalysis:
    """Analysis of existing indexes."""
    index_name: str
    table_name: str
    columns: List[str]
    index_type: str
    is_unique: bool
    is_primary: bool
    usage_count: int
    last_used: Optional[datetime]
    size_mb: float
    is_redundant: bool
    is_unused: bool
    recommendations: List[str]


class IndexAdvisorService:
    """Service for analyzing and recommending database indexes."""
    
    def __init__(self, db_type: str = "postgresql"):
        self.db_type = db_type
        self.analyzer = QueryAnalyzerService()
    
    async def analyze_query_for_indexes(
        self,
        session: AsyncSession,
        query: str,
        execution_plan: Optional[Dict[str, Any]] = None
    ) -> List[IndexRecommendation]:
        """
        Analyze a query and recommend indexes.
        
        Args:
            session: Database session
            query: SQL query to analyze
            execution_plan: Optional pre-computed execution plan
            
        Returns:
            List of index recommendations
        """
        analysis = self.analyzer.analyze(query)
        recommendations = []
        
        # Analyze WHERE clause columns
        where_indexes = self._analyze_where_clause(analysis)
        recommendations.extend(where_indexes)
        
        # Analyze JOIN columns
        join_indexes = self._analyze_joins(analysis)
        recommendations.extend(join_indexes)
        
        # Analyze ORDER BY columns
        order_indexes = self._analyze_order_by(analysis)
        recommendations.extend(order_indexes)
        
        # Analyze GROUP BY columns
        group_indexes = self._analyze_group_by(analysis)
        recommendations.extend(group_indexes)
        
        # Check execution plan for seq scans
        if execution_plan:
            plan_indexes = self._analyze_execution_plan(execution_plan, analysis)
            recommendations.extend(plan_indexes)
        
        # Deduplicate and prioritize
        recommendations = self._deduplicate_recommendations(recommendations)
        recommendations = self._prioritize_recommendations(recommendations)
        
        # Check for existing indexes
        existing = await self.get_existing_indexes(session)
        recommendations = self._filter_existing_indexes(recommendations, existing)
        
        return recommendations
    
    def _analyze_where_clause(
        self, 
        analysis: QueryAnalysis
    ) -> List[IndexRecommendation]:
        """Analyze WHERE clause for index opportunities."""
        recommendations = []
        
        for condition in analysis.where_conditions:
            # Extract column from condition (simplified parsing)
            columns = self._extract_columns_from_condition(condition)
            
            for table, column in columns:
                if table and column:
                    rec = self._create_recommendation(
                        table_name=table,
                        columns=[column],
                        reason="Column used in WHERE clause",
                        priority=8
                    )
                    recommendations.append(rec)
        
        return recommendations
    
    def _analyze_joins(
        self, 
        analysis: QueryAnalysis
    ) -> List[IndexRecommendation]:
        """Analyze JOIN conditions for index opportunities."""
        recommendations = []
        
        for join in analysis.joins:
            # Parse join condition to extract columns
            condition = join.get('condition', '')
            table = join.get('table', '')
            
            # Extract columns from join condition
            columns = self._extract_columns_from_condition(condition)
            
            for _, column in columns:
                if column:
                    rec = self._create_recommendation(
                        table_name=table,
                        columns=[column],
                        reason="Column used in JOIN condition",
                        priority=9
                    )
                    recommendations.append(rec)
        
        return recommendations
    
    def _analyze_order_by(
        self, 
        analysis: QueryAnalysis
    ) -> List[IndexRecommendation]:
        """Analyze ORDER BY for index opportunities."""
        recommendations = []
        
        if analysis.order_by:
            # Extract table.column or just column
            columns = []
            for order_col in analysis.order_by:
                # Remove ASC/DESC
                col = order_col.split()[0].strip()
                if '.' in col:
                    table, column = col.split('.', 1)
                else:
                    table = analysis.tables[0] if analysis.tables else None
                    column = col
                if table and column:
                    columns.append(column)
            
            if columns and analysis.tables:
                rec = self._create_recommendation(
                    table_name=analysis.tables[0],
                    columns=columns,
                    reason="Columns used in ORDER BY clause",
                    priority=6
                )
                recommendations.append(rec)
        
        return recommendations
    
    def _analyze_group_by(
        self, 
        analysis: QueryAnalysis
    ) -> List[IndexRecommendation]:
        """Analyze GROUP BY for index opportunities."""
        recommendations = []
        
        if analysis.group_by:
            columns = []
            for group_col in analysis.group_by:
                col = group_col.strip()
                if '.' in col:
                    _, column = col.split('.', 1)
                else:
                    column = col
                columns.append(column)
            
            if columns and analysis.tables:
                rec = self._create_recommendation(
                    table_name=analysis.tables[0],
                    columns=columns,
                    reason="Columns used in GROUP BY clause",
                    priority=7
                )
                recommendations.append(rec)
        
        return recommendations
    
    def _analyze_execution_plan(
        self,
        execution_plan: Dict[str, Any],
        analysis: QueryAnalysis
    ) -> List[IndexRecommendation]:
        """Analyze execution plan for sequential scans."""
        recommendations = []
        
        plan_data = execution_plan.get("plan", {})
        
        # Look for Seq Scan nodes in PostgreSQL plans
        if execution_plan.get("db_type") == "postgresql":
            seq_scans = self._find_seq_scans_postgresql(plan_data)
            for scan in seq_scans:
                table = scan.get("Relation Name")
                filter_cond = scan.get("Filter", "")
                
                if table and filter_cond:
                    columns = self._extract_columns_from_condition(filter_cond)
                    for _, col in columns:
                        if col:
                            rec = self._create_recommendation(
                                table_name=table,
                                columns=[col],
                                reason="Sequential scan detected on filtered column",
                                priority=10
                            )
                            recommendations.append(rec)
        
        return recommendations
    
    def _find_seq_scans_postgresql(
        self, 
        plan: Any, 
        scans: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Recursively find Seq Scan nodes in PostgreSQL plan."""
        if scans is None:
            scans = []
        
        if isinstance(plan, list):
            for item in plan:
                self._find_seq_scans_postgresql(item, scans)
        elif isinstance(plan, dict):
            if plan.get("Node Type") == "Seq Scan":
                scans.append(plan)
            
            # Check child plans
            for key in ["Plans", "Plan"]:
                if key in plan:
                    self._find_seq_scans_postgresql(plan[key], scans)
        
        return scans
    
    def _extract_columns_from_condition(
        self, 
        condition: str
    ) -> List[Tuple[Optional[str], str]]:
        """Extract table.column pairs from a condition string."""
        import re
        
        columns = []
        
        # Pattern: table.column or just column followed by operator
        pattern = r'(\w+)\.(\w+)|(?<![.\w])(\w+)(?=\s*[=<>!]|\s+(?:IN|LIKE|BETWEEN|IS))'
        
        matches = re.findall(pattern, condition, re.IGNORECASE)
        
        for match in matches:
            if match[0] and match[1]:  # table.column
                columns.append((match[0], match[1]))
            elif match[2]:  # just column
                columns.append((None, match[2]))
        
        return columns
    
    def _create_recommendation(
        self,
        table_name: str,
        columns: List[str],
        reason: str,
        priority: int,
        index_type: str = "btree"
    ) -> IndexRecommendation:
        """Create an index recommendation."""
        # Generate index name
        col_names = "_".join(columns)[:30]
        index_name = f"idx_{table_name}_{col_names}"
        
        # Generate SQL
        columns_str = ", ".join(columns)
        
        if self.db_type == "postgresql":
            create_sql = f"CREATE INDEX {index_name} ON {table_name} USING {index_type} ({columns_str})"
        else:
            create_sql = f"CREATE INDEX {index_name} ON {table_name} ({columns_str})"
        
        drop_sql = f"DROP INDEX {index_name}"
        
        return IndexRecommendation(
            table_name=table_name,
            column_names=columns,
            index_type=index_type,
            index_name=index_name,
            create_statement=create_sql,
            drop_statement=drop_sql,
            estimated_improvement_ms=0,  # Would be calculated by ML model
            estimated_storage_mb=0,  # Would be estimated
            confidence=0.7,  # Base confidence
            reasoning=reason,
            affected_queries=[],
            priority=priority
        )
    
    def _deduplicate_recommendations(
        self, 
        recommendations: List[IndexRecommendation]
    ) -> List[IndexRecommendation]:
        """Remove duplicate recommendations."""
        seen = set()
        unique = []
        
        for rec in recommendations:
            key = (rec.table_name, tuple(sorted(rec.column_names)))
            if key not in seen:
                seen.add(key)
                unique.append(rec)
        
        return unique
    
    def _prioritize_recommendations(
        self, 
        recommendations: List[IndexRecommendation]
    ) -> List[IndexRecommendation]:
        """Sort recommendations by priority."""
        return sorted(recommendations, key=lambda x: x.priority, reverse=True)
    
    def _filter_existing_indexes(
        self,
        recommendations: List[IndexRecommendation],
        existing: List[IndexAnalysis]
    ) -> List[IndexRecommendation]:
        """Remove recommendations for indexes that already exist."""
        existing_set = set()
        for idx in existing:
            key = (idx.table_name, tuple(sorted(idx.columns)))
            existing_set.add(key)
        
        filtered = []
        for rec in recommendations:
            key = (rec.table_name, tuple(sorted(rec.column_names)))
            if key not in existing_set:
                filtered.append(rec)
        
        return filtered
    
    async def get_existing_indexes(
        self, 
        session: AsyncSession,
        table_name: Optional[str] = None
    ) -> List[IndexAnalysis]:
        """Get existing indexes from the database."""
        if self.db_type == "postgresql":
            return await self._get_indexes_postgresql(session, table_name)
        else:
            return await self._get_indexes_mysql(session, table_name)
    
    async def _get_indexes_postgresql(
        self,
        session: AsyncSession,
        table_name: Optional[str] = None
    ) -> List[IndexAnalysis]:
        """Get indexes from PostgreSQL."""
        query = text("""
            SELECT
                i.relname as index_name,
                t.relname as table_name,
                array_agg(a.attname ORDER BY x.n) as columns,
                am.amname as index_type,
                ix.indisunique as is_unique,
                ix.indisprimary as is_primary,
                pg_relation_size(i.oid) as size_bytes,
                s.idx_scan as usage_count
            FROM pg_index ix
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_am am ON am.oid = i.relam
            JOIN pg_namespace n ON n.oid = t.relnamespace
            LEFT JOIN pg_stat_user_indexes s ON s.indexrelid = i.oid
            CROSS JOIN LATERAL unnest(ix.indkey) WITH ORDINALITY AS x(attnum, n)
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = x.attnum
            WHERE n.nspname = 'public'
            AND (:table_name IS NULL OR t.relname = :table_name)
            GROUP BY i.relname, t.relname, am.amname, ix.indisunique, 
                     ix.indisprimary, i.oid, s.idx_scan
            ORDER BY t.relname, i.relname
        """)
        
        try:
            result = await session.execute(query, {"table_name": table_name})
            rows = result.fetchall()
            
            indexes = []
            for row in rows:
                indexes.append(IndexAnalysis(
                    index_name=row[0],
                    table_name=row[1],
                    columns=list(row[2]) if row[2] else [],
                    index_type=row[3],
                    is_unique=row[4],
                    is_primary=row[5],
                    usage_count=row[7] or 0,
                    last_used=None,
                    size_mb=row[6] / (1024 * 1024) if row[6] else 0,
                    is_redundant=False,
                    is_unused=row[7] == 0 if row[7] is not None else False,
                    recommendations=[]
                ))
            return indexes
        except Exception as e:
            return []
    
    async def _get_indexes_mysql(
        self,
        session: AsyncSession,
        table_name: Optional[str] = None
    ) -> List[IndexAnalysis]:
        """Get indexes from MySQL."""
        query = text("""
            SELECT
                INDEX_NAME as index_name,
                TABLE_NAME as table_name,
                GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) as columns,
                INDEX_TYPE as index_type,
                NON_UNIQUE = 0 as is_unique,
                INDEX_NAME = 'PRIMARY' as is_primary
            FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE()
            AND (:table_name IS NULL OR TABLE_NAME = :table_name)
            GROUP BY INDEX_NAME, TABLE_NAME, INDEX_TYPE, NON_UNIQUE
            ORDER BY TABLE_NAME, INDEX_NAME
        """)
        
        try:
            result = await session.execute(query, {"table_name": table_name})
            rows = result.fetchall()
            
            indexes = []
            for row in rows:
                indexes.append(IndexAnalysis(
                    index_name=row[0],
                    table_name=row[1],
                    columns=row[2].split(',') if row[2] else [],
                    index_type=row[3],
                    is_unique=row[4],
                    is_primary=row[5],
                    usage_count=0,
                    last_used=None,
                    size_mb=0,
                    is_redundant=False,
                    is_unused=False,
                    recommendations=[]
                ))
            return indexes
        except Exception as e:
            return []
    
    async def find_unused_indexes(
        self,
        session: AsyncSession,
        min_days_unused: int = 30
    ) -> List[IndexAnalysis]:
        """Find indexes that haven't been used."""
        indexes = await self.get_existing_indexes(session)
        
        unused = []
        for idx in indexes:
            if idx.is_unused and not idx.is_primary and not idx.is_unique:
                idx.recommendations.append(
                    f"Index appears unused - consider dropping to save {idx.size_mb:.2f} MB"
                )
                unused.append(idx)
        
        return unused
    
    async def find_redundant_indexes(
        self,
        session: AsyncSession
    ) -> List[IndexAnalysis]:
        """Find redundant indexes (covered by other indexes)."""
        indexes = await self.get_existing_indexes(session)
        
        # Group by table
        by_table: Dict[str, List[IndexAnalysis]] = {}
        for idx in indexes:
            if idx.table_name not in by_table:
                by_table[idx.table_name] = []
            by_table[idx.table_name].append(idx)
        
        redundant = []
        
        for table, table_indexes in by_table.items():
            for i, idx1 in enumerate(table_indexes):
                for idx2 in table_indexes[i+1:]:
                    # Check if idx1 is a prefix of idx2
                    if self._is_prefix_of(idx1.columns, idx2.columns):
                        idx1.is_redundant = True
                        idx1.recommendations.append(
                            f"Index is covered by {idx2.index_name}"
                        )
                        redundant.append(idx1)
                    elif self._is_prefix_of(idx2.columns, idx1.columns):
                        idx2.is_redundant = True
                        idx2.recommendations.append(
                            f"Index is covered by {idx1.index_name}"
                        )
                        redundant.append(idx2)
        
        return redundant
    
    def _is_prefix_of(self, cols1: List[str], cols2: List[str]) -> bool:
        """Check if cols1 is a prefix of cols2."""
        if len(cols1) >= len(cols2):
            return False
        return cols2[:len(cols1)] == cols1
    
    async def estimate_index_impact(
        self,
        session: AsyncSession,
        recommendation: IndexRecommendation,
        sample_queries: List[str]
    ) -> Dict[str, Any]:
        """Estimate the impact of creating an index."""
        # This would involve creating a hypothetical index analysis
        # For now, return estimated metrics
        
        return {
            "estimated_improvement_percentage": 30,
            "affected_query_count": len(sample_queries),
            "estimated_storage_mb": 10,
            "creation_time_estimate": "< 1 minute",
            "recommendation": "Create index" if len(sample_queries) > 5 else "Low priority"
        }
