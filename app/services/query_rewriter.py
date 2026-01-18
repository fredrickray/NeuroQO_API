"""
Query Rewriter Service - Automatically rewrites queries for optimization.
"""
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from app.services.query_analyzer import QueryAnalyzerService, QueryAnalysis


class RewriteRule(str, Enum):
    """Types of query rewrite rules."""
    SELECT_STAR_TO_COLUMNS = "select_star_to_columns"
    OR_TO_IN = "or_to_in"
    OR_TO_UNION = "or_to_union"
    NOT_IN_TO_NOT_EXISTS = "not_in_to_not_exists"
    SUBQUERY_TO_JOIN = "subquery_to_join"
    ADD_LIMIT = "add_limit"
    REMOVE_DISTINCT = "remove_distinct"
    OPTIMIZE_LIKE = "optimize_like"
    REORDER_JOINS = "reorder_joins"
    PUSH_DOWN_PREDICATES = "push_down_predicates"


@dataclass
class RewriteResult:
    """Result of a query rewrite attempt."""
    original_query: str
    rewritten_query: str
    rules_applied: List[RewriteRule]
    confidence: float  # 0-1 confidence that rewrite is beneficial
    estimated_improvement: str  # Description of expected improvement
    warnings: List[str]


class QueryRewriterService:
    """Service for automatically rewriting SQL queries for optimization."""
    
    def __init__(self, db_type: str = "postgresql"):
        self.db_type = db_type
        self.analyzer = QueryAnalyzerService()
    
    def rewrite(
        self, 
        query: str,
        table_columns: Optional[Dict[str, List[str]]] = None,
        apply_rules: Optional[List[RewriteRule]] = None
    ) -> RewriteResult:
        """
        Rewrite a query applying optimization rules.
        
        Args:
            query: Original SQL query
            table_columns: Dict mapping table names to their columns
            apply_rules: Specific rules to apply (None = all applicable)
            
        Returns:
            RewriteResult with rewritten query and metadata
        """
        analysis = self.analyzer.analyze(query)
        
        rewritten = query
        rules_applied = []
        warnings = []
        confidence = 1.0
        improvements = []
        
        # Define all rewrite methods
        rewrite_methods = {
            RewriteRule.SELECT_STAR_TO_COLUMNS: self._rewrite_select_star,
            RewriteRule.OR_TO_IN: self._rewrite_or_to_in,
            RewriteRule.OR_TO_UNION: self._rewrite_or_to_union,
            RewriteRule.NOT_IN_TO_NOT_EXISTS: self._rewrite_not_in_to_not_exists,
            RewriteRule.SUBQUERY_TO_JOIN: self._rewrite_subquery_to_join,
            RewriteRule.ADD_LIMIT: self._rewrite_add_limit,
            RewriteRule.OPTIMIZE_LIKE: self._rewrite_optimize_like,
            RewriteRule.PUSH_DOWN_PREDICATES: self._rewrite_push_down_predicates,
        }
        
        # Apply each rule
        for rule, method in rewrite_methods.items():
            if apply_rules and rule not in apply_rules:
                continue
            
            try:
                result = method(rewritten, analysis, table_columns)
                if result:
                    new_query, improvement, rule_confidence, rule_warnings = result
                    if new_query != rewritten:
                        rewritten = new_query
                        rules_applied.append(rule)
                        improvements.append(improvement)
                        confidence = min(confidence, rule_confidence)
                        warnings.extend(rule_warnings)
            except Exception as e:
                warnings.append(f"Error applying {rule.value}: {str(e)}")
        
        return RewriteResult(
            original_query=query,
            rewritten_query=rewritten,
            rules_applied=rules_applied,
            confidence=confidence,
            estimated_improvement="; ".join(improvements) if improvements else "No changes",
            warnings=warnings
        )
    
    def _rewrite_select_star(
        self, 
        query: str, 
        analysis: QueryAnalysis,
        table_columns: Optional[Dict[str, List[str]]] = None
    ) -> Optional[Tuple[str, str, float, List[str]]]:
        """Replace SELECT * with explicit column names."""
        if 'SELECT *' not in query.upper() and 'SELECT  *' not in query.upper():
            return None
        
        if not table_columns or not analysis.tables:
            return None
        
        # Get columns for all tables in the query
        columns = []
        for table in analysis.tables:
            if table in table_columns:
                for col in table_columns[table]:
                    columns.append(f"{table}.{col}")
        
        if not columns:
            return None
        
        # Replace SELECT * with column list
        column_list = ", ".join(columns)
        pattern = r'SELECT\s+\*'
        rewritten = re.sub(pattern, f'SELECT {column_list}', query, flags=re.IGNORECASE)
        
        return (
            rewritten,
            "Replaced SELECT * with explicit columns for better performance",
            0.9,
            ["Ensure all required columns are included"]
        )
    
    def _rewrite_or_to_in(
        self, 
        query: str, 
        analysis: QueryAnalysis,
        table_columns: Optional[Dict[str, List[str]]] = None
    ) -> Optional[Tuple[str, str, float, List[str]]]:
        """Convert multiple OR conditions on same column to IN clause."""
        # Pattern to find: column = 'value1' OR column = 'value2' ...
        pattern = r"(\w+)\s*=\s*'([^']+)'\s+OR\s+\1\s*=\s*'([^']+)'(?:\s+OR\s+\1\s*=\s*'([^']+)')*"
        
        def replace_or_with_in(match):
            column = match.group(1)
            # Extract all values
            values = re.findall(r"=\s*'([^']+)'", match.group(0))
            in_values = ", ".join(f"'{v}'" for v in values)
            return f"{column} IN ({in_values})"
        
        rewritten = re.sub(pattern, replace_or_with_in, query, flags=re.IGNORECASE)
        
        if rewritten != query:
            return (
                rewritten,
                "Converted OR conditions to IN clause for better index utilization",
                0.95,
                []
            )
        
        return None
    
    def _rewrite_or_to_union(
        self, 
        query: str, 
        analysis: QueryAnalysis,
        table_columns: Optional[Dict[str, List[str]]] = None
    ) -> Optional[Tuple[str, str, float, List[str]]]:
        """Convert OR conditions on different columns to UNION for better index usage."""
        # This is a more complex rewrite - only apply to specific patterns
        # Pattern: WHERE (col1 = val1 OR col2 = val2) on different indexed columns
        
        # Check if there are OR conditions on different columns
        or_match = re.search(
            r'WHERE\s+(?:\()?(\w+)\s*=\s*[^)]+\s+OR\s+(\w+)\s*=',
            query, 
            re.IGNORECASE
        )
        
        if not or_match:
            return None
        
        col1, col2 = or_match.group(1), or_match.group(2)
        
        # Only rewrite if columns are different
        if col1.lower() == col2.lower():
            return None
        
        # This rewrite is complex and risky - return with low confidence
        # In production, we'd need more sophisticated parsing
        warnings = [
            "UNION rewrite suggested - verify semantics are preserved",
            "Consider if both columns are indexed before applying"
        ]
        
        # Don't actually rewrite - just suggest
        return None  # Return None to skip this complex rewrite
    
    def _rewrite_not_in_to_not_exists(
        self, 
        query: str, 
        analysis: QueryAnalysis,
        table_columns: Optional[Dict[str, List[str]]] = None
    ) -> Optional[Tuple[str, str, float, List[str]]]:
        """Convert NOT IN with subquery to NOT EXISTS for better performance."""
        # Pattern: WHERE column NOT IN (SELECT ...)
        pattern = r'(\w+)\s+NOT\s+IN\s*\(\s*SELECT\s+(\w+)\s+FROM\s+(\w+)(?:\s+WHERE\s+([^)]+))?\s*\)'
        
        match = re.search(pattern, query, re.IGNORECASE | re.DOTALL)
        
        if not match:
            return None
        
        outer_col = match.group(1)
        inner_col = match.group(2)
        inner_table = match.group(3)
        inner_where = match.group(4)
        
        # Build NOT EXISTS clause
        not_exists = f"NOT EXISTS (SELECT 1 FROM {inner_table} WHERE {inner_table}.{inner_col} = {outer_col}"
        if inner_where:
            not_exists += f" AND {inner_where}"
        not_exists += ")"
        
        rewritten = re.sub(pattern, not_exists, query, flags=re.IGNORECASE | re.DOTALL)
        
        return (
            rewritten,
            "Converted NOT IN to NOT EXISTS for better performance with NULLs",
            0.85,
            ["Verify NULL handling matches expected behavior"]
        )
    
    def _rewrite_subquery_to_join(
        self, 
        query: str, 
        analysis: QueryAnalysis,
        table_columns: Optional[Dict[str, List[str]]] = None
    ) -> Optional[Tuple[str, str, float, List[str]]]:
        """Convert correlated subquery in SELECT to JOIN."""
        # Pattern: SELECT (SELECT col FROM table WHERE condition) ...
        # This is a complex transformation that requires careful analysis
        
        if not analysis.has_subquery:
            return None
        
        # Check for scalar subquery in SELECT clause
        select_subquery_pattern = r'SELECT\s+.*?\(\s*SELECT\s+'
        
        if not re.search(select_subquery_pattern, query, re.IGNORECASE):
            return None
        
        # This transformation is risky - flag for manual review
        return None  # Skip automatic rewrite, but could be flagged as suggestion
    
    def _rewrite_add_limit(
        self, 
        query: str, 
        analysis: QueryAnalysis,
        table_columns: Optional[Dict[str, List[str]]] = None
    ) -> Optional[Tuple[str, str, float, List[str]]]:
        """Add LIMIT clause to queries without one."""
        if analysis.has_limit:
            return None
        
        if analysis.query_type.value != "SELECT":
            return None
        
        # Don't add LIMIT to aggregate queries
        if analysis.has_aggregation and not analysis.group_by:
            return None
        
        # Add a default limit
        default_limit = 1000
        
        # Clean up the query and add LIMIT
        query_stripped = query.rstrip().rstrip(';')
        rewritten = f"{query_stripped} LIMIT {default_limit}"
        
        return (
            rewritten,
            f"Added LIMIT {default_limit} to prevent unbounded result sets",
            0.7,
            ["Verify that limiting results is acceptable for this query"]
        )
    
    def _rewrite_optimize_like(
        self, 
        query: str, 
        analysis: QueryAnalysis,
        table_columns: Optional[Dict[str, List[str]]] = None
    ) -> Optional[Tuple[str, str, float, List[str]]]:
        """Optimize LIKE patterns where possible."""
        # Check for leading wildcard that could be avoided
        # Pattern: LIKE '%value'
        
        # We can't automatically fix leading wildcards, but we can flag them
        if re.search(r"LIKE\s+'%[^%]", query, re.IGNORECASE):
            # Can't rewrite, but issue is already flagged in analysis
            return None
        
        # Check for LIKE 'exact_value' without wildcards - convert to =
        pattern = r"(\w+)\s+LIKE\s+'([^%_]+)'"
        
        def replace_like_with_equals(match):
            column = match.group(1)
            value = match.group(2)
            return f"{column} = '{value}'"
        
        rewritten = re.sub(pattern, replace_like_with_equals, query, flags=re.IGNORECASE)
        
        if rewritten != query:
            return (
                rewritten,
                "Converted LIKE without wildcards to = for better index usage",
                0.95,
                []
            )
        
        return None
    
    def _rewrite_push_down_predicates(
        self, 
        query: str, 
        analysis: QueryAnalysis,
        table_columns: Optional[Dict[str, List[str]]] = None
    ) -> Optional[Tuple[str, str, float, List[str]]]:
        """Push WHERE predicates into subqueries or derived tables."""
        # This is a complex optimization that the database usually handles
        # We'll skip automatic rewriting but could flag for review
        return None
    
    def suggest_rewrites(self, query: str) -> List[Dict[str, Any]]:
        """
        Get suggestions for query rewrites without applying them.
        
        Args:
            query: SQL query to analyze
            
        Returns:
            List of suggested rewrites with explanations
        """
        analysis = self.analyzer.analyze(query)
        suggestions = []
        
        # Check for SELECT *
        if 'SELECT *' in query.upper():
            suggestions.append({
                "rule": RewriteRule.SELECT_STAR_TO_COLUMNS.value,
                "description": "Replace SELECT * with explicit column names",
                "impact": "high",
                "reason": "Reduces data transfer and allows better query planning"
            })
        
        # Check for OR conditions
        if ' OR ' in query.upper():
            suggestions.append({
                "rule": RewriteRule.OR_TO_IN.value,
                "description": "Consider converting OR conditions to IN clause",
                "impact": "medium",
                "reason": "IN clauses can be optimized better by the query planner"
            })
        
        # Check for NOT IN
        if 'NOT IN' in query.upper() and 'SELECT' in query.upper():
            suggestions.append({
                "rule": RewriteRule.NOT_IN_TO_NOT_EXISTS.value,
                "description": "Convert NOT IN subquery to NOT EXISTS",
                "impact": "high",
                "reason": "NOT EXISTS handles NULLs better and often performs faster"
            })
        
        # Check for missing LIMIT
        if not analysis.has_limit and analysis.query_type.value == "SELECT":
            suggestions.append({
                "rule": RewriteRule.ADD_LIMIT.value,
                "description": "Add LIMIT clause to prevent unbounded results",
                "impact": "medium",
                "reason": "Prevents accidental large result sets"
            })
        
        # Check for subqueries
        if analysis.has_subquery:
            suggestions.append({
                "rule": RewriteRule.SUBQUERY_TO_JOIN.value,
                "description": "Consider converting subquery to JOIN",
                "impact": "high",
                "reason": "JOINs are often more efficient than subqueries"
            })
        
        # Check for LIKE with leading wildcard
        if re.search(r"LIKE\s+'%", query, re.IGNORECASE):
            suggestions.append({
                "rule": "avoid_leading_wildcard",
                "description": "Avoid leading wildcard in LIKE patterns",
                "impact": "high",
                "reason": "Leading wildcards prevent index usage"
            })
        
        return suggestions
    
    def validate_rewrite(
        self, 
        original: str, 
        rewritten: str
    ) -> Dict[str, Any]:
        """
        Validate that a rewritten query is semantically equivalent.
        
        Args:
            original: Original query
            rewritten: Rewritten query
            
        Returns:
            Validation result with comparison details
        """
        orig_analysis = self.analyzer.analyze(original)
        new_analysis = self.analyzer.analyze(rewritten)
        
        issues = []
        
        # Check that query type is preserved
        if orig_analysis.query_type != new_analysis.query_type:
            issues.append("Query type changed")
        
        # Check that tables are preserved
        if set(orig_analysis.tables) != set(new_analysis.tables):
            issues.append(f"Tables changed: {orig_analysis.tables} -> {new_analysis.tables}")
        
        # Check for potential semantic differences
        if orig_analysis.has_distinct != new_analysis.has_distinct:
            issues.append("DISTINCT clause changed")
        
        if orig_analysis.has_aggregation != new_analysis.has_aggregation:
            issues.append("Aggregation changed")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "original_complexity": orig_analysis.estimated_complexity,
            "new_complexity": new_analysis.estimated_complexity,
            "complexity_improved": self._complexity_score(new_analysis.estimated_complexity) < 
                                   self._complexity_score(orig_analysis.estimated_complexity)
        }
    
    def _complexity_score(self, complexity: str) -> int:
        """Convert complexity string to numeric score."""
        scores = {
            "simple": 1,
            "moderate": 2,
            "complex": 3,
            "very_complex": 4,
            "unknown": 2
        }
        return scores.get(complexity, 2)
