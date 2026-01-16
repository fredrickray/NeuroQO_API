"""
Optimization Recommender - ML-based query optimization recommendations.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from app.ml.feature_extractor import QueryFeatureExtractor, QueryFeatures
from app.services.query_analyzer import QueryAnalyzerService


class OptimizationType(str, Enum):
    """Types of query optimizations."""
    INDEX_SUGGESTION = "index_suggestion"
    QUERY_REWRITE = "query_rewrite"
    SELECT_OPTIMIZATION = "select_optimization"
    JOIN_OPTIMIZATION = "join_optimization"
    SUBQUERY_ELIMINATION = "subquery_elimination"
    PREDICATE_PUSHDOWN = "predicate_pushdown"
    CACHING = "caching"
    LIMIT_ADDITION = "limit_addition"
    DISTINCT_REMOVAL = "distinct_removal"
    OR_TO_IN = "or_to_in"
    NOT_IN_TO_EXISTS = "not_in_to_exists"
    WILDCARD_OPTIMIZATION = "wildcard_optimization"


@dataclass
class OptimizationSuggestion:
    """A single optimization suggestion."""
    optimization_type: OptimizationType
    confidence: float
    description: str
    implementation_hint: str
    estimated_improvement_pct: float
    risk_level: str  # low, medium, high
    priority: int  # 1-10


@dataclass
class OptimizationRecommendations:
    """Complete recommendations for a query."""
    query_hash: str
    suggestions: List[OptimizationSuggestion]
    overall_optimization_score: float  # 0-100
    estimated_total_improvement_pct: float
    model_confidence: float
    query_issues: List[str]


class OptimizationRecommender:
    """
    ML-based system for recommending query optimizations.
    
    Uses multi-label classification to recommend multiple
    optimization strategies for a single query.
    """
    
    # Optimization rules and their triggers
    OPTIMIZATION_RULES = {
        OptimizationType.SELECT_OPTIMIZATION: {
            "trigger_features": ["has_select_star"],
            "description": "Replace SELECT * with specific columns",
            "risk": "low",
            "avg_improvement": 15
        },
        OptimizationType.INDEX_SUGGESTION: {
            "trigger_features": ["where_condition_count", "join_count"],
            "description": "Add index on filtered/joined columns",
            "risk": "low",
            "avg_improvement": 40
        },
        OptimizationType.JOIN_OPTIMIZATION: {
            "trigger_features": ["join_count", "cross_join_count", "has_implicit_join"],
            "description": "Optimize join order or type",
            "risk": "medium",
            "avg_improvement": 25
        },
        OptimizationType.SUBQUERY_ELIMINATION: {
            "trigger_features": ["has_subquery", "subquery_depth"],
            "description": "Convert subquery to JOIN",
            "risk": "medium",
            "avg_improvement": 30
        },
        OptimizationType.OR_TO_IN: {
            "trigger_features": ["has_or_condition", "or_count"],
            "description": "Convert OR conditions to IN clause",
            "risk": "low",
            "avg_improvement": 10
        },
        OptimizationType.NOT_IN_TO_EXISTS: {
            "trigger_features": ["has_not_in"],
            "description": "Convert NOT IN to NOT EXISTS",
            "risk": "low",
            "avg_improvement": 20
        },
        OptimizationType.WILDCARD_OPTIMIZATION: {
            "trigger_features": ["has_leading_wildcard"],
            "description": "Optimize LIKE pattern usage",
            "risk": "high",
            "avg_improvement": 50
        },
        OptimizationType.CACHING: {
            "trigger_features": ["aggregation_count", "group_by_count"],
            "description": "Cache query results",
            "risk": "low",
            "avg_improvement": 80
        },
        OptimizationType.LIMIT_ADDITION: {
            "trigger_features": ["has_limit"],
            "description": "Add LIMIT clause to restrict results",
            "risk": "low",
            "avg_improvement": 30
        },
        OptimizationType.DISTINCT_REMOVAL: {
            "trigger_features": ["has_distinct"],
            "description": "Remove unnecessary DISTINCT",
            "risk": "medium",
            "avg_improvement": 15
        },
        OptimizationType.PREDICATE_PUSHDOWN: {
            "trigger_features": ["has_subquery", "where_condition_count"],
            "description": "Push predicates into subqueries",
            "risk": "medium",
            "avg_improvement": 20
        }
    }
    
    def __init__(self, model_path: Optional[str] = None):
        self.feature_extractor = QueryFeatureExtractor()
        self.analyzer = QueryAnalyzerService()
        
        # ML model for optimization prediction
        self.model: Optional[RandomForestClassifier] = None
        self.scaler = StandardScaler()
        self.label_binarizer = MultiLabelBinarizer()
        
        # Model metadata
        self.model_version = "1.0.0"
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def train(
        self,
        queries: List[str],
        applied_optimizations: List[List[OptimizationType]],
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the recommendation model.
        
        Args:
            queries: List of SQL queries
            applied_optimizations: List of optimization types that were effective
            test_size: Fraction of data for testing
            
        Returns:
            Training metrics
        """
        # Check minimum sample requirement
        min_samples = 5
        if len(queries) < min_samples:
            raise ValueError(
                f"Insufficient training data: {len(queries)} samples provided, "
                f"but at least {min_samples} are required. "
                "Add more optimization results to train the recommender model."
            )
        
        # Extract features
        X, _ = self.feature_extractor.extract_batch(queries)
        
        # Encode labels (multi-label)
        y = self.label_binarizer.fit_transform([
            [opt.value for opt in opts] for opts in applied_optimizations
        ])
        
        # Check if there are any valid labels
        if y.shape[1] == 0:
            raise ValueError(
                "No valid optimization labels found in training data. "
                "Ensure optimization_rules_applied contains valid OptimizationType values: "
                f"{[e.value for e in OptimizationType]}"
            )
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        metrics = {
            "f1_micro": f1_score(y_test, y_pred, average='micro'),
            "f1_macro": f1_score(y_test, y_pred, average='macro'),
            "precision_micro": precision_score(y_test, y_pred, average='micro'),
            "recall_micro": recall_score(y_test, y_pred, average='micro'),
        }
        
        self.is_trained = True
        return metrics
    
    def recommend(self, query: str) -> OptimizationRecommendations:
        """
        Generate optimization recommendations for a query.
        
        Args:
            query: SQL query to analyze
            
        Returns:
            OptimizationRecommendations with all suggestions
        """
        features = self.feature_extractor.extract(query)
        analysis = self.analyzer.analyze(query)
        
        suggestions = []
        
        # If model is trained, use ML predictions
        if self.is_trained and self.model is not None:
            ml_suggestions = self._ml_recommend(features)
            suggestions.extend(ml_suggestions)
        
        # Always apply rule-based recommendations
        rule_suggestions = self._rule_based_recommend(features, analysis)
        
        # Merge suggestions, avoiding duplicates
        seen_types = {s.optimization_type for s in suggestions}
        for suggestion in rule_suggestions:
            if suggestion.optimization_type not in seen_types:
                suggestions.append(suggestion)
                seen_types.add(suggestion.optimization_type)
        
        # Sort by priority and confidence
        suggestions.sort(key=lambda x: (x.priority, x.confidence), reverse=True)
        
        # Identify query issues
        issues = self._identify_issues(features, analysis)
        
        # Calculate overall metrics
        total_improvement = sum(s.estimated_improvement_pct for s in suggestions[:5])
        optimization_score = min(100, total_improvement)
        avg_confidence = np.mean([s.confidence for s in suggestions]) if suggestions else 0
        
        return OptimizationRecommendations(
            query_hash=features.query_hash,
            suggestions=suggestions[:10],  # Top 10 suggestions
            overall_optimization_score=optimization_score,
            estimated_total_improvement_pct=total_improvement,
            model_confidence=float(avg_confidence),
            query_issues=issues
        )
    
    def _ml_recommend(
        self, 
        features: QueryFeatures
    ) -> List[OptimizationSuggestion]:
        """Generate recommendations using ML model."""
        if self.model is None:
            return []
        
        X = self.scaler.transform(features.feature_vector.reshape(1, -1))
        
        # Get prediction probabilities
        probas = self.model.predict_proba(X)  # type: ignore[union-attr]
        
        suggestions = []
        
        for i, (classes, proba) in enumerate(zip(self.model.classes_, probas)):  # type: ignore[union-attr]
            if len(proba.shape) > 1:
                confidence = proba[0][1] if proba.shape[1] > 1 else proba[0][0]
            else:
                confidence = proba[0]
            
            if confidence > 0.3:  # Threshold
                try:
                    opt_type = OptimizationType(self.label_binarizer.classes_[i])
                    rule = self.OPTIMIZATION_RULES.get(opt_type, {})
                    
                    suggestions.append(OptimizationSuggestion(
                        optimization_type=opt_type,
                        confidence=float(confidence),
                        description=rule.get("description", ""),
                        implementation_hint=self._get_implementation_hint(opt_type, features),
                        estimated_improvement_pct=rule.get("avg_improvement", 10),
                        risk_level=rule.get("risk", "medium"),
                        priority=int(confidence * 10)
                    ))
                except (ValueError, KeyError):
                    continue
        
        return suggestions
    
    def _rule_based_recommend(
        self, 
        features: QueryFeatures,
        analysis: Any
    ) -> List[OptimizationSuggestion]:
        """Generate rule-based recommendations."""
        suggestions = []
        fd = features.feature_dict
        
        # SELECT * optimization
        if fd.get("has_select_star", 0):
            suggestions.append(OptimizationSuggestion(
                optimization_type=OptimizationType.SELECT_OPTIMIZATION,
                confidence=0.95,
                description="Replace SELECT * with explicit column list",
                implementation_hint="Specify only the columns you need to reduce data transfer",
                estimated_improvement_pct=15,
                risk_level="low",
                priority=8
            ))
        
        # Index suggestions based on WHERE/JOIN
        if fd.get("where_condition_count", 0) > 0 or fd.get("join_count", 0) > 0:
            suggestions.append(OptimizationSuggestion(
                optimization_type=OptimizationType.INDEX_SUGGESTION,
                confidence=0.85,
                description="Consider adding indexes on filtered/joined columns",
                implementation_hint=f"Analyze columns in WHERE clause and JOIN conditions: {analysis.where_conditions[:2]}",
                estimated_improvement_pct=40,
                risk_level="low",
                priority=9
            ))
        
        # Join optimization
        if fd.get("join_count", 0) >= 3 or fd.get("has_implicit_join", 0):
            suggestions.append(OptimizationSuggestion(
                optimization_type=OptimizationType.JOIN_OPTIMIZATION,
                confidence=0.8,
                description="Optimize join order and use explicit JOIN syntax",
                implementation_hint="Reorder joins to filter data early, use INNER/LEFT JOIN explicitly",
                estimated_improvement_pct=25,
                risk_level="medium",
                priority=7
            ))
        
        # Subquery optimization
        if fd.get("has_subquery", 0) and fd.get("subquery_depth", 0) >= 1:
            suggestions.append(OptimizationSuggestion(
                optimization_type=OptimizationType.SUBQUERY_ELIMINATION,
                confidence=0.75,
                description="Consider converting subquery to JOIN",
                implementation_hint="Rewrite correlated subqueries as JOINs for better optimization",
                estimated_improvement_pct=30,
                risk_level="medium",
                priority=7
            ))
        
        # OR to IN conversion
        if fd.get("or_count", 0) >= 2:
            suggestions.append(OptimizationSuggestion(
                optimization_type=OptimizationType.OR_TO_IN,
                confidence=0.9,
                description="Convert multiple OR conditions to IN clause",
                implementation_hint="Replace 'col = a OR col = b OR col = c' with 'col IN (a, b, c)'",
                estimated_improvement_pct=10,
                risk_level="low",
                priority=6
            ))
        
        # NOT IN to NOT EXISTS
        if fd.get("has_not_in", 0):
            suggestions.append(OptimizationSuggestion(
                optimization_type=OptimizationType.NOT_IN_TO_EXISTS,
                confidence=0.9,
                description="Convert NOT IN subquery to NOT EXISTS",
                implementation_hint="NOT EXISTS handles NULLs correctly and often performs better",
                estimated_improvement_pct=20,
                risk_level="low",
                priority=8
            ))
        
        # Leading wildcard warning
        if fd.get("has_leading_wildcard", 0):
            suggestions.append(OptimizationSuggestion(
                optimization_type=OptimizationType.WILDCARD_OPTIMIZATION,
                confidence=0.95,
                description="Avoid leading wildcard in LIKE patterns",
                implementation_hint="Leading '%' prevents index usage. Consider full-text search or restructuring",
                estimated_improvement_pct=50,
                risk_level="high",
                priority=9
            ))
        
        # Caching for aggregation queries
        if fd.get("has_aggregation", 0) and fd.get("group_by_count", 0) > 0:
            suggestions.append(OptimizationSuggestion(
                optimization_type=OptimizationType.CACHING,
                confidence=0.7,
                description="Consider caching aggregation results",
                implementation_hint="Aggregate queries on stable data are good caching candidates",
                estimated_improvement_pct=80,
                risk_level="low",
                priority=5
            ))
        
        # Add LIMIT
        if not fd.get("has_limit", 0) and fd.get("table_count", 0) > 0:
            if not fd.get("has_aggregation", 0) or fd.get("group_by_count", 0) > 0:
                suggestions.append(OptimizationSuggestion(
                    optimization_type=OptimizationType.LIMIT_ADDITION,
                    confidence=0.6,
                    description="Consider adding LIMIT clause",
                    implementation_hint="Prevent unbounded result sets with appropriate LIMIT",
                    estimated_improvement_pct=30,
                    risk_level="low",
                    priority=4
                ))
        
        return suggestions
    
    def _get_implementation_hint(
        self, 
        opt_type: OptimizationType,
        features: QueryFeatures
    ) -> str:
        """Get specific implementation hint based on query features."""
        fd = features.feature_dict
        
        hints = {
            OptimizationType.INDEX_SUGGESTION: f"Create indexes on columns used in WHERE/JOIN",
            OptimizationType.SELECT_OPTIMIZATION: "List specific columns instead of SELECT *",
            OptimizationType.JOIN_OPTIMIZATION: f"Optimize {int(fd.get('join_count', 0))} joins",
            OptimizationType.SUBQUERY_ELIMINATION: "Convert to JOIN for better execution plan",
            OptimizationType.OR_TO_IN: f"Convert {int(fd.get('or_count', 0))} OR conditions to IN",
            OptimizationType.NOT_IN_TO_EXISTS: "Use NOT EXISTS for better NULL handling",
            OptimizationType.WILDCARD_OPTIMIZATION: "Avoid leading wildcards in LIKE",
            OptimizationType.CACHING: "Cache results for repeated aggregate queries",
            OptimizationType.LIMIT_ADDITION: "Add LIMIT to prevent large result sets",
            OptimizationType.DISTINCT_REMOVAL: "Verify DISTINCT is necessary",
            OptimizationType.PREDICATE_PUSHDOWN: "Push WHERE conditions into subqueries"
        }
        
        return hints.get(opt_type, "Review query structure")
    
    def _identify_issues(
        self, 
        features: QueryFeatures,
        analysis: Any
    ) -> List[str]:
        """Identify issues in the query."""
        issues = []
        fd = features.feature_dict
        
        if fd.get("has_select_star", 0):
            issues.append("Uses SELECT * - specify columns explicitly")
        
        if fd.get("has_leading_wildcard", 0):
            issues.append("Leading wildcard in LIKE prevents index usage")
        
        if fd.get("has_function_in_where", 0):
            issues.append("Function on column in WHERE may prevent index usage")
        
        if fd.get("has_implicit_join", 0):
            issues.append("Uses implicit join syntax (comma-separated tables)")
        
        if fd.get("has_not_in", 0):
            issues.append("NOT IN with subquery may have NULL handling issues")
        
        if fd.get("cross_join_count", 0) > 0:
            issues.append("Contains CROSS JOIN - verify this is intentional")
        
        if fd.get("subquery_depth", 0) > 2:
            issues.append("Deeply nested subqueries may impact performance")
        
        if not fd.get("has_limit", 0) and fd.get("estimated_complexity_score", 0) > 10:
            issues.append("Complex query without LIMIT clause")
        
        return issues
    
    def save(self, path: str):
        """Save the model to disk."""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_binarizer": self.label_binarizer,
            "model_version": self.model_version,
            "is_trained": self.is_trained
        }
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, path: str):
        """Load the model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.label_binarizer = model_data["label_binarizer"]
        self.model_version = model_data["model_version"]
        self.is_trained = model_data["is_trained"]
    
    def recommend_batch(
        self, 
        queries: List[str]
    ) -> List[OptimizationRecommendations]:
        """Generate recommendations for multiple queries."""
        return [self.recommend(q) for q in queries]
