"""
Feature Extractor - Extracts ML features from SQL queries.
"""
import re
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from app.services.query_analyzer import QueryAnalyzerService, QueryAnalysis


class FeatureCategory(str, Enum):
    """Categories of features."""
    STRUCTURAL = "structural"
    COMPLEXITY = "complexity"
    PATTERN = "pattern"
    STATISTICAL = "statistical"


@dataclass
class QueryFeatures:
    """Extracted features from a SQL query."""
    query_hash: str
    feature_vector: np.ndarray
    feature_names: List[str]
    feature_dict: Dict[str, float]
    category_features: Dict[FeatureCategory, Dict[str, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "query_hash": self.query_hash,
            "features": self.feature_dict,
            "vector": self.feature_vector.tolist()
        }


class QueryFeatureExtractor:
    """
    Extracts numerical features from SQL queries for ML models.
    
    Features include:
    - Structural features (table count, join count, etc.)
    - Complexity features (subqueries, aggregations, etc.)
    - Pattern features (presence of specific SQL constructs)
    - Statistical features (query length, token count, etc.)
    """
    
    # SQL keywords for feature extraction
    SQL_KEYWORDS = {
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER',
        'FULL', 'CROSS', 'ON', 'AND', 'OR', 'NOT', 'IN', 'EXISTS', 'BETWEEN',
        'LIKE', 'IS', 'NULL', 'GROUP', 'BY', 'HAVING', 'ORDER', 'ASC', 'DESC',
        'LIMIT', 'OFFSET', 'UNION', 'INTERSECT', 'EXCEPT', 'DISTINCT', 'ALL',
        'AS', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'CAST', 'COALESCE'
    }
    
    AGGREGATE_FUNCTIONS = {'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'ARRAY_AGG', 'STRING_AGG'}
    
    WINDOW_FUNCTIONS = {'ROW_NUMBER', 'RANK', 'DENSE_RANK', 'NTILE', 'LAG', 'LEAD', 'FIRST_VALUE', 'LAST_VALUE'}
    
    def __init__(self):
        self.analyzer = QueryAnalyzerService()
        self._feature_names: List[str] = []
        self._initialize_feature_names()
    
    def _initialize_feature_names(self):
        """Initialize the list of feature names."""
        self._feature_names = [
            # Structural features
            "table_count",
            "column_count",
            "join_count",
            "inner_join_count",
            "left_join_count",
            "right_join_count",
            "cross_join_count",
            "where_condition_count",
            "and_count",
            "or_count",
            "order_by_count",
            "group_by_count",
            
            # Complexity features
            "has_subquery",
            "subquery_depth",
            "has_aggregation",
            "aggregation_count",
            "has_window_function",
            "window_function_count",
            "has_distinct",
            "has_limit",
            "has_offset",
            "has_union",
            "has_case_when",
            "has_cte",
            
            # Pattern features
            "has_select_star",
            "has_not_in",
            "has_not_exists",
            "has_like_wildcard",
            "has_leading_wildcard",
            "has_or_condition",
            "has_in_clause",
            "has_between",
            "has_is_null",
            "has_function_in_where",
            "has_implicit_join",
            
            # Statistical features
            "query_length",
            "token_count",
            "keyword_count",
            "identifier_count",
            "literal_count",
            "operator_count",
            "parenthesis_depth",
            "avg_identifier_length",
            
            # Complexity scores
            "estimated_complexity_score",
            "join_complexity_score",
            "predicate_complexity_score"
        ]
    
    def extract(self, query: str) -> QueryFeatures:
        """
        Extract all features from a SQL query.
        
        Args:
            query: SQL query string
            
        Returns:
            QueryFeatures object containing all extracted features
        """
        # Get basic analysis
        analysis = self.analyzer.analyze(query)
        
        # Extract features by category
        structural = self._extract_structural_features(query, analysis)
        complexity = self._extract_complexity_features(query, analysis)
        pattern = self._extract_pattern_features(query, analysis)
        statistical = self._extract_statistical_features(query)
        
        # Combine all features
        feature_dict = {}
        feature_dict.update(structural)
        feature_dict.update(complexity)
        feature_dict.update(pattern)
        feature_dict.update(statistical)
        
        # Add complexity scores
        feature_dict.update(self._calculate_complexity_scores(feature_dict))
        
        # Create feature vector in consistent order
        feature_vector = np.array([
            feature_dict.get(name, 0) for name in self._feature_names
        ], dtype=np.float32)
        
        return QueryFeatures(
            query_hash=analysis.query_hash,
            feature_vector=feature_vector,
            feature_names=self._feature_names.copy(),
            feature_dict=feature_dict,
            category_features={
                FeatureCategory.STRUCTURAL: structural,
                FeatureCategory.COMPLEXITY: complexity,
                FeatureCategory.PATTERN: pattern,
                FeatureCategory.STATISTICAL: statistical
            }
        )
    
    def _extract_structural_features(
        self, 
        query: str, 
        analysis: QueryAnalysis
    ) -> Dict[str, float]:
        """Extract structural features from the query."""
        query_upper = query.upper()
        
        # Count different join types
        inner_joins = len(re.findall(r'\bINNER\s+JOIN\b', query_upper))
        left_joins = len(re.findall(r'\bLEFT\s+(?:OUTER\s+)?JOIN\b', query_upper))
        right_joins = len(re.findall(r'\bRIGHT\s+(?:OUTER\s+)?JOIN\b', query_upper))
        cross_joins = len(re.findall(r'\bCROSS\s+JOIN\b', query_upper))
        
        # Count total joins (including implicit)
        total_joins = len(analysis.joins)
        if inner_joins + left_joins + right_joins + cross_joins == 0:
            inner_joins = total_joins  # Assume INNER for plain JOIN
        
        # Count AND/OR in WHERE clause
        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|$)', query_upper, re.DOTALL)
        where_clause = where_match.group(1) if where_match else ""
        
        and_count = len(re.findall(r'\bAND\b', where_clause))
        or_count = len(re.findall(r'\bOR\b', where_clause))
        
        return {
            "table_count": float(len(analysis.tables)),
            "column_count": float(len(analysis.columns)),
            "join_count": float(total_joins),
            "inner_join_count": float(inner_joins),
            "left_join_count": float(left_joins),
            "right_join_count": float(right_joins),
            "cross_join_count": float(cross_joins),
            "where_condition_count": float(len(analysis.where_conditions)),
            "and_count": float(and_count),
            "or_count": float(or_count),
            "order_by_count": float(len(analysis.order_by)),
            "group_by_count": float(len(analysis.group_by))
        }
    
    def _extract_complexity_features(
        self, 
        query: str, 
        analysis: QueryAnalysis
    ) -> Dict[str, float]:
        """Extract complexity-related features."""
        query_upper = query.upper()
        
        # Subquery analysis
        subquery_depth = self._calculate_subquery_depth(query)
        
        # Count aggregations
        agg_count = sum(
            len(re.findall(rf'\b{func}\s*\(', query_upper)) 
            for func in self.AGGREGATE_FUNCTIONS
        )
        
        # Count window functions
        window_count = sum(
            len(re.findall(rf'\b{func}\s*\(', query_upper)) 
            for func in self.WINDOW_FUNCTIONS
        )
        
        # Check for CTE (WITH clause)
        has_cte = query_upper.strip().startswith('WITH')
        
        return {
            "has_subquery": float(analysis.has_subquery),
            "subquery_depth": float(subquery_depth),
            "has_aggregation": float(analysis.has_aggregation),
            "aggregation_count": float(agg_count),
            "has_window_function": float(window_count > 0),
            "window_function_count": float(window_count),
            "has_distinct": float(analysis.has_distinct),
            "has_limit": float(analysis.has_limit),
            "has_offset": float('OFFSET' in query_upper),
            "has_union": float('UNION' in query_upper),
            "has_case_when": float('CASE' in query_upper and 'WHEN' in query_upper),
            "has_cte": float(has_cte)
        }
    
    def _extract_pattern_features(
        self, 
        query: str, 
        analysis: QueryAnalysis
    ) -> Dict[str, float]:
        """Extract SQL anti-pattern and pattern features."""
        query_upper = query.upper()
        
        # Check for SELECT *
        has_select_star = bool(re.search(r'SELECT\s+\*', query_upper))
        
        # Check for NOT IN with subquery
        has_not_in = bool(re.search(r'NOT\s+IN\s*\(\s*SELECT', query_upper))
        
        # Check for NOT EXISTS
        has_not_exists = bool(re.search(r'NOT\s+EXISTS', query_upper))
        
        # Check for LIKE patterns
        has_like = 'LIKE' in query_upper
        has_leading_wildcard = bool(re.search(r"LIKE\s+'%", query_upper))
        
        # Check for function in WHERE clause
        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|$)', query_upper, re.DOTALL)
        where_clause = where_match.group(1) if where_match else ""
        has_function_in_where = bool(re.search(r'\w+\s*\([^)]+\)\s*[=<>]', where_clause))
        
        # Check for implicit join (comma-separated tables)
        has_implicit_join = bool(re.search(r'FROM\s+\w+\s*,\s*\w+', query_upper))
        
        return {
            "has_select_star": float(has_select_star),
            "has_not_in": float(has_not_in),
            "has_not_exists": float(has_not_exists),
            "has_like_wildcard": float(has_like),
            "has_leading_wildcard": float(has_leading_wildcard),
            "has_or_condition": float(' OR ' in query_upper),
            "has_in_clause": float(' IN ' in query_upper or ' IN(' in query_upper),
            "has_between": float('BETWEEN' in query_upper),
            "has_is_null": float('IS NULL' in query_upper or 'IS NOT NULL' in query_upper),
            "has_function_in_where": float(has_function_in_where),
            "has_implicit_join": float(has_implicit_join)
        }
    
    def _extract_statistical_features(self, query: str) -> Dict[str, float]:
        """Extract statistical features from the query."""
        query_upper = query.upper()
        
        # Tokenize (simple split)
        tokens = re.findall(r'\w+|[^\w\s]', query)
        
        # Count different token types
        keywords = [t for t in tokens if t.upper() in self.SQL_KEYWORDS]
        identifiers = [t for t in tokens if t.upper() not in self.SQL_KEYWORDS and re.match(r'^\w+$', t)]
        literals = re.findall(r"'[^']*'|\d+", query)
        operators = re.findall(r'[=<>!]+|AND|OR|NOT|IN|LIKE|BETWEEN', query_upper)
        
        # Calculate parenthesis depth
        max_depth = 0
        current_depth = 0
        for char in query:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        
        # Average identifier length
        avg_id_len = np.mean([len(i) for i in identifiers]) if identifiers else 0
        
        return {
            "query_length": float(len(query)),
            "token_count": float(len(tokens)),
            "keyword_count": float(len(keywords)),
            "identifier_count": float(len(identifiers)),
            "literal_count": float(len(literals)),
            "operator_count": float(len(operators)),
            "parenthesis_depth": float(max_depth),
            "avg_identifier_length": float(avg_id_len)
        }
    
    def _calculate_subquery_depth(self, query: str) -> int:
        """Calculate the maximum depth of nested subqueries."""
        depth = 0
        max_depth = 0
        in_select = False
        
        query_upper = query.upper()
        i = 0
        while i < len(query_upper):
            if query_upper[i:i+6] == 'SELECT':
                if in_select:
                    depth += 1
                    max_depth = max(max_depth, depth)
                in_select = True
                i += 6
            elif query_upper[i] == ')' and depth > 0:
                depth -= 1
                i += 1
            else:
                i += 1
        
        return max_depth
    
    def _calculate_complexity_scores(
        self, 
        features: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate composite complexity scores."""
        # Overall complexity score
        complexity_score = (
            features.get("table_count", 0) * 2 +
            features.get("join_count", 0) * 3 +
            features.get("subquery_depth", 0) * 5 +
            features.get("aggregation_count", 0) * 2 +
            features.get("window_function_count", 0) * 3 +
            features.get("where_condition_count", 0) * 1 +
            features.get("has_cte", 0) * 4
        )
        
        # Join complexity
        join_complexity = (
            features.get("join_count", 0) * 2 +
            features.get("cross_join_count", 0) * 5 +
            features.get("has_implicit_join", 0) * 3
        )
        
        # Predicate complexity
        predicate_complexity = (
            features.get("where_condition_count", 0) +
            features.get("or_count", 0) * 2 +
            features.get("has_function_in_where", 0) * 3 +
            features.get("has_leading_wildcard", 0) * 4 +
            features.get("has_not_in", 0) * 3
        )
        
        return {
            "estimated_complexity_score": float(complexity_score),
            "join_complexity_score": float(join_complexity),
            "predicate_complexity_score": float(predicate_complexity)
        }
    
    def get_feature_names(self) -> List[str]:
        """Get the list of feature names in order."""
        return self._feature_names.copy()
    
    def extract_batch(self, queries: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from multiple queries.
        
        Args:
            queries: List of SQL queries
            
        Returns:
            Tuple of (feature matrix, list of query hashes)
        """
        features_list = []
        hashes = []
        
        for query in queries:
            features = self.extract(query)
            features_list.append(features.feature_vector)
            hashes.append(features.query_hash)
        
        return np.array(features_list), hashes
