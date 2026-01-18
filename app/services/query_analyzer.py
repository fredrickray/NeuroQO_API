"""
Query Analyzer Service - Parses, normalizes, and analyzes SQL queries.
"""
import re
import hashlib
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Parenthesis
from sqlparse.tokens import Keyword, DML
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class QueryType(str, Enum):
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    OTHER = "OTHER"


@dataclass
class QueryAnalysis:
    """Result of query analysis."""
    query_hash: str
    normalized_query: str
    query_type: QueryType
    tables: List[str]
    columns: List[str]
    joins: List[Dict[str, str]]
    where_conditions: List[str]
    order_by: List[str]
    group_by: List[str]
    has_subquery: bool
    has_aggregation: bool
    has_distinct: bool
    has_limit: bool
    estimated_complexity: str
    potential_issues: List[str]


class QueryAnalyzerService:
    """Service for analyzing and parsing SQL queries."""
    
    # Patterns for query normalization
    LITERAL_PATTERNS = [
        (r"'[^']*'", "'?'"),  # String literals
        (r"\b\d+\b", "?"),     # Numeric literals
        (r"\b0x[0-9a-fA-F]+\b", "?"),  # Hex literals
    ]
    
    # Aggregate functions
    AGGREGATE_FUNCTIONS = {'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'GROUP_CONCAT', 'ARRAY_AGG'}
    
    # Known anti-patterns
    ANTI_PATTERNS = {
        'SELECT *': 'Avoid SELECT * - specify columns explicitly',
        'SELECT DISTINCT': 'DISTINCT can be expensive - ensure it\'s necessary',
        'ORDER BY RAND()': 'ORDER BY RAND() is very slow - consider alternatives',
        'LIKE \'%': 'Leading wildcard in LIKE prevents index usage',
        'NOT IN': 'NOT IN can be slow - consider NOT EXISTS or LEFT JOIN',
        'OR': 'Multiple OR conditions may prevent index usage',
        '!= NULL': 'Use IS NOT NULL instead of != NULL',
        '= NULL': 'Use IS NULL instead of = NULL',
    }
    
    def __init__(self):
        self.parsed_cache: Dict[str, QueryAnalysis] = {}
    
    def analyze(self, query: str) -> QueryAnalysis:
        """
        Perform comprehensive analysis of a SQL query.
        
        Args:
            query: The SQL query to analyze
            
        Returns:
            QueryAnalysis object with all analysis results
        """
        # Normalize whitespace
        query = ' '.join(query.split())
        
        # Check cache
        query_hash = self._compute_hash(query)
        if query_hash in self.parsed_cache:
            return self.parsed_cache[query_hash]
        
        # Parse the query
        parsed = sqlparse.parse(query)[0] if sqlparse.parse(query) else None
        
        if not parsed:
            return self._empty_analysis(query_hash, query)
        
        # Extract components
        query_type = self._get_query_type(parsed)
        tables = self._extract_tables(parsed)
        columns = self._extract_columns(parsed)
        joins = self._extract_joins(query)
        where_conditions = self._extract_where_conditions(parsed)
        order_by = self._extract_order_by(query)
        group_by = self._extract_group_by(query)
        
        # Analyze characteristics
        has_subquery = self._has_subquery(query)
        has_aggregation = self._has_aggregation(query)
        has_distinct = 'DISTINCT' in query.upper()
        has_limit = 'LIMIT' in query.upper()
        
        # Compute complexity
        complexity = self._estimate_complexity(
            tables, joins, has_subquery, has_aggregation, where_conditions
        )
        
        # Find potential issues
        issues = self._find_issues(query)
        
        # Normalize query (replace literals with placeholders)
        normalized = self._normalize_query(query)
        
        analysis = QueryAnalysis(
            query_hash=query_hash,
            normalized_query=normalized,
            query_type=query_type,
            tables=tables,
            columns=columns,
            joins=joins,
            where_conditions=where_conditions,
            order_by=order_by,
            group_by=group_by,
            has_subquery=has_subquery,
            has_aggregation=has_aggregation,
            has_distinct=has_distinct,
            has_limit=has_limit,
            estimated_complexity=complexity,
            potential_issues=issues
        )
        
        # Cache the result
        self.parsed_cache[query_hash] = analysis
        
        return analysis
    
    def _compute_hash(self, query: str) -> str:
        """Compute MD5 hash of the query."""
        normalized = self._normalize_query(query)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query by replacing literals with placeholders."""
        normalized = query
        for pattern, replacement in self.LITERAL_PATTERNS:
            normalized = re.sub(pattern, replacement, normalized)
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        return normalized.upper()
    
    def _get_query_type(self, parsed) -> QueryType:
        """Determine the type of SQL query."""
        for token in parsed.tokens:
            if token.ttype is DML:
                token_value = token.value.upper()
                if token_value == 'SELECT':
                    return QueryType.SELECT
                elif token_value == 'INSERT':
                    return QueryType.INSERT
                elif token_value == 'UPDATE':
                    return QueryType.UPDATE
                elif token_value == 'DELETE':
                    return QueryType.DELETE
        return QueryType.OTHER
    
    def _extract_tables(self, parsed) -> List[str]:
        """Extract table names from the query."""
        tables = []
        from_seen = False
        
        for token in parsed.tokens:
            if from_seen:
                if self._is_subselect(token):
                    tables.extend(self._extract_from_subselect(token))
                elif token.ttype is Keyword:
                    from_seen = False
                else:
                    tables.extend(self._extract_table_identifiers(token))
            elif token.ttype is Keyword and token.value.upper() == 'FROM':
                from_seen = True
        
        # Also extract from JOIN clauses
        query_str = str(parsed)
        join_pattern = r'(?:INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN\s+(\w+)'
        join_tables = re.findall(join_pattern, query_str, re.IGNORECASE)
        tables.extend(join_tables)
        
        return list(set(tables))
    
    def _extract_table_identifiers(self, token) -> List[str]:
        """Extract table names from a token."""
        tables = []
        if isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                name = self._get_table_name(identifier)
                if name:
                    tables.append(name)
        elif isinstance(token, Identifier):
            name = self._get_table_name(token)
            if name:
                tables.append(name)
        elif token.ttype is not Keyword:
            name = token.value.strip()
            if name and not name.startswith('('):
                # Remove alias
                name = name.split()[0] if name.split() else name
                tables.append(name)
        return tables
    
    def _get_table_name(self, identifier) -> Optional[str]:
        """Get table name from an identifier."""
        if hasattr(identifier, 'get_real_name'):
            return identifier.get_real_name()
        return str(identifier).split()[0] if str(identifier).split() else None
    
    def _extract_columns(self, parsed) -> List[str]:
        """Extract column names from SELECT clause."""
        columns = []
        select_seen = False
        
        for token in parsed.tokens:
            if select_seen:
                if token.ttype is Keyword and token.value.upper() == 'FROM':
                    break
                if isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        columns.append(str(identifier).strip())
                elif isinstance(token, Identifier):
                    columns.append(str(token).strip())
                elif not token.is_whitespace:
                    col = str(token).strip()
                    if col and col != ',':
                        columns.append(col)
            elif token.ttype is DML and token.value.upper() == 'SELECT':
                select_seen = True
        
        return columns
    
    def _extract_joins(self, query: str) -> List[Dict[str, str]]:
        """Extract JOIN information from the query."""
        joins = []
        
        # Pattern for different join types
        join_pattern = r'(INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?\s+ON\s+([^JOIN]+?)(?=(?:INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN|\s*WHERE|\s*GROUP|\s*ORDER|\s*LIMIT|$)'
        
        matches = re.findall(join_pattern, query, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            join_type, table, alias, condition = match
            joins.append({
                'type': join_type.upper() if join_type else 'INNER',
                'table': table,
                'alias': alias or table,
                'condition': condition.strip()
            })
        
        return joins
    
    def _extract_where_conditions(self, parsed) -> List[str]:
        """Extract WHERE conditions."""
        conditions = []
        
        for token in parsed.tokens:
            if isinstance(token, Where):
                # Get the condition part (skip 'WHERE' keyword)
                condition_str = str(token)[5:].strip()
                # Split by AND/OR
                parts = re.split(r'\s+AND\s+|\s+OR\s+', condition_str, flags=re.IGNORECASE)
                conditions.extend([p.strip() for p in parts if p.strip()])
        
        return conditions
    
    def _extract_order_by(self, query: str) -> List[str]:
        """Extract ORDER BY columns."""
        match = re.search(r'ORDER\s+BY\s+(.+?)(?:LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
        if match:
            order_str = match.group(1).strip()
            return [col.strip() for col in order_str.split(',')]
        return []
    
    def _extract_group_by(self, query: str) -> List[str]:
        """Extract GROUP BY columns."""
        match = re.search(r'GROUP\s+BY\s+(.+?)(?:HAVING|ORDER|LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
        if match:
            group_str = match.group(1).strip()
            return [col.strip() for col in group_str.split(',')]
        return []
    
    def _has_subquery(self, query: str) -> bool:
        """Check if query contains a subquery."""
        # Count SELECT keywords (more than 1 means subquery)
        select_count = len(re.findall(r'\bSELECT\b', query, re.IGNORECASE))
        return select_count > 1
    
    def _has_aggregation(self, query: str) -> bool:
        """Check if query uses aggregate functions."""
        query_upper = query.upper()
        return any(f'{func}(' in query_upper for func in self.AGGREGATE_FUNCTIONS)
    
    def _is_subselect(self, token) -> bool:
        """Check if token is a subselect."""
        if isinstance(token, Parenthesis):
            for sub_token in token.tokens:
                if sub_token.ttype is DML and sub_token.value.upper() == 'SELECT':
                    return True
        return False
    
    def _extract_from_subselect(self, token) -> List[str]:
        """Extract tables from a subselect."""
        # Recursively parse the subselect
        inner_query = str(token)[1:-1]  # Remove parentheses
        parsed = sqlparse.parse(inner_query)[0] if sqlparse.parse(inner_query) else None
        if parsed:
            return self._extract_tables(parsed)
        return []
    
    def _estimate_complexity(
        self, 
        tables: List[str], 
        joins: List[Dict], 
        has_subquery: bool,
        has_aggregation: bool,
        where_conditions: List[str]
    ) -> str:
        """Estimate query complexity."""
        score = 0
        
        # Table count
        score += len(tables) * 2
        
        # Join count
        score += len(joins) * 3
        
        # Subquery
        if has_subquery:
            score += 5
        
        # Aggregation
        if has_aggregation:
            score += 2
        
        # WHERE conditions
        score += len(where_conditions)
        
        if score <= 3:
            return "simple"
        elif score <= 7:
            return "moderate"
        elif score <= 12:
            return "complex"
        else:
            return "very_complex"
    
    def _find_issues(self, query: str) -> List[str]:
        """Find potential performance issues in the query."""
        issues = []
        query_upper = query.upper()
        
        for pattern, message in self.ANTI_PATTERNS.items():
            if pattern in query_upper:
                issues.append(message)
        
        # Check for functions on indexed columns in WHERE
        if re.search(r'WHERE\s+\w+\s*\(', query, re.IGNORECASE):
            issues.append('Function on column in WHERE may prevent index usage')
        
        # Check for implicit type conversion
        if re.search(r"=\s*'\d+'", query):
            issues.append('Possible implicit type conversion (string to number)')
        
        return issues
    
    def _empty_analysis(self, query_hash: str, query: str) -> QueryAnalysis:
        """Return empty analysis for unparseable queries."""
        return QueryAnalysis(
            query_hash=query_hash,
            normalized_query=query,
            query_type=QueryType.OTHER,
            tables=[],
            columns=[],
            joins=[],
            where_conditions=[],
            order_by=[],
            group_by=[],
            has_subquery=False,
            has_aggregation=False,
            has_distinct=False,
            has_limit=False,
            estimated_complexity="unknown",
            potential_issues=[]
        )
    
    def get_query_fingerprint(self, query: str) -> str:
        """Get a fingerprint for query pattern matching."""
        return self._compute_hash(query)
    
    def compare_queries(self, query1: str, query2: str) -> Dict[str, Any]:
        """Compare two queries and return differences."""
        analysis1 = self.analyze(query1)
        analysis2 = self.analyze(query2)
        
        return {
            'same_pattern': analysis1.query_hash == analysis2.query_hash,
            'same_tables': set(analysis1.tables) == set(analysis2.tables),
            'same_type': analysis1.query_type == analysis2.query_type,
            'complexity_change': f"{analysis1.estimated_complexity} -> {analysis2.estimated_complexity}",
            'tables_added': list(set(analysis2.tables) - set(analysis1.tables)),
            'tables_removed': list(set(analysis1.tables) - set(analysis2.tables)),
        }
