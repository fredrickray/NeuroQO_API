"""
Query Classifier - Classifies queries for optimization routing.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from app.ml.feature_extractor import QueryFeatureExtractor, QueryFeatures


class QueryCategory(str, Enum):
    """Categories of queries for optimization routing."""
    SIMPLE_LOOKUP = "simple_lookup"  # Simple WHERE clause lookups
    AGGREGATION = "aggregation"  # Queries with GROUP BY
    JOIN_HEAVY = "join_heavy"  # Multiple table joins
    ANALYTICAL = "analytical"  # Complex analytical queries
    REPORTING = "reporting"  # Large result set reports
    TRANSACTIONAL = "transactional"  # OLTP-style queries


class OptimizationPriority(str, Enum):
    """Priority levels for optimization."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SKIP = "skip"


@dataclass
class ClassificationResult:
    """Result of query classification."""
    query_hash: str
    category: QueryCategory
    category_confidence: float
    optimization_priority: OptimizationPriority
    priority_confidence: float
    optimization_types: List[str]  # Suggested optimization types
    reasoning: str


class QueryClassifier:
    """
    ML-based classifier for categorizing queries and determining optimization priority.
    
    Uses Random Forest for multi-class classification.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.feature_extractor = QueryFeatureExtractor()
        
        # Models
        self.category_model: Optional[RandomForestClassifier] = None
        self.priority_model: Optional[RandomForestClassifier] = None
        
        # Preprocessors
        self.scaler = StandardScaler()
        self.category_encoder = LabelEncoder()
        self.priority_encoder = LabelEncoder()
        
        # Model metadata
        self.model_version = "1.0.0"
        self.is_trained = False
        
        # Load models if path provided
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def train(
        self,
        queries: List[str],
        categories: List[QueryCategory],
        priorities: List[OptimizationPriority],
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the classifier models.
        
        Args:
            queries: List of SQL queries
            categories: Corresponding category labels
            priorities: Corresponding priority labels
            test_size: Fraction of data to use for testing
            
        Returns:
            Training metrics
        """
        # Extract features
        X, _ = self.feature_extractor.extract_batch(queries)
        
        # Encode labels
        y_category = self.category_encoder.fit_transform([c.value for c in categories])
        y_priority = self.priority_encoder.fit_transform([p.value for p in priorities])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
            X_scaled, y_category, y_priority, test_size=test_size, random_state=42
        )
        
        # Train category classifier
        self.category_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.category_model.fit(X_train, y_cat_train)
        
        # Train priority classifier
        self.priority_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.priority_model.fit(X_train, y_pri_train)
        
        # Evaluate
        cat_pred = self.category_model.predict(X_test)
        pri_pred = self.priority_model.predict(X_test)
        
        # Get the unique labels present in both training and test data
        cat_labels = np.arange(len(self.category_encoder.classes_))
        pri_labels = np.arange(len(self.priority_encoder.classes_))
        
        metrics = {
            "category_accuracy": accuracy_score(y_cat_test, cat_pred),
            "priority_accuracy": accuracy_score(y_pri_test, pri_pred),
            "category_report": classification_report(
                y_cat_test, cat_pred, 
                labels=cat_labels,
                target_names=self.category_encoder.classes_,
                output_dict=True,
                zero_division=0
            ),
            "priority_report": classification_report(
                y_pri_test, pri_pred,
                labels=pri_labels,
                target_names=self.priority_encoder.classes_,
                output_dict=True,
                zero_division=0
            ),
            "feature_importance": dict(zip(
                self.feature_extractor.get_feature_names(),
                self.category_model.feature_importances_
            ))
        }
        
        self.is_trained = True
        return metrics
    
    def classify(self, query: str) -> ClassificationResult:
        """
        Classify a single query.
        
        Args:
            query: SQL query to classify
            
        Returns:
            ClassificationResult with category and priority
        """
        features = self.feature_extractor.extract(query)
        
        # Use rule-based classification if model not trained
        if not self.is_trained:
            return self._rule_based_classify(query, features)
        
        # Scale features
        X = self.scaler.transform(features.feature_vector.reshape(1, -1))
        
        # Safety check - should not happen due to is_trained check above
        if self.category_model is None or self.priority_model is None:
            return self._rule_based_classify(query, features)
        
        # Predict category
        cat_proba = self.category_model.predict_proba(X)[0]
        cat_idx = np.argmax(cat_proba)
        category = QueryCategory(self.category_encoder.classes_[cat_idx])
        cat_confidence = cat_proba[cat_idx]
        
        # Predict priority
        pri_proba = self.priority_model.predict_proba(X)[0]
        pri_idx = np.argmax(pri_proba)
        priority = OptimizationPriority(self.priority_encoder.classes_[pri_idx])
        pri_confidence = pri_proba[pri_idx]
        
        # Determine optimization types
        opt_types = self._suggest_optimization_types(features, category)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(features, category, priority)
        
        return ClassificationResult(
            query_hash=features.query_hash,
            category=category,
            category_confidence=cat_confidence,
            optimization_priority=priority,
            priority_confidence=pri_confidence,
            optimization_types=opt_types,
            reasoning=reasoning
        )
    
    def _rule_based_classify(
        self, 
        query: str, 
        features: QueryFeatures
    ) -> ClassificationResult:
        """Rule-based classification when ML model is not trained."""
        fd = features.feature_dict
        
        # Determine category based on features
        if fd.get("has_aggregation", 0) and fd.get("group_by_count", 0) > 0:
            category = QueryCategory.AGGREGATION
        elif fd.get("join_count", 0) >= 3:
            category = QueryCategory.JOIN_HEAVY
        elif fd.get("has_window_function", 0) or fd.get("has_cte", 0):
            category = QueryCategory.ANALYTICAL
        elif fd.get("table_count", 0) == 1 and fd.get("where_condition_count", 0) <= 2:
            category = QueryCategory.SIMPLE_LOOKUP
        elif not fd.get("has_limit", 0) and fd.get("table_count", 0) > 1:
            category = QueryCategory.REPORTING
        else:
            category = QueryCategory.TRANSACTIONAL
        
        # Determine priority based on complexity
        complexity = fd.get("estimated_complexity_score", 0)
        if complexity > 20:
            priority = OptimizationPriority.CRITICAL
        elif complexity > 15:
            priority = OptimizationPriority.HIGH
        elif complexity > 8:
            priority = OptimizationPriority.MEDIUM
        elif complexity > 3:
            priority = OptimizationPriority.LOW
        else:
            priority = OptimizationPriority.SKIP
        
        # Check for anti-patterns to boost priority
        if fd.get("has_select_star", 0) or fd.get("has_leading_wildcard", 0):
            if priority == OptimizationPriority.LOW:
                priority = OptimizationPriority.MEDIUM
            elif priority == OptimizationPriority.MEDIUM:
                priority = OptimizationPriority.HIGH
        
        opt_types = self._suggest_optimization_types(features, category)
        reasoning = self._generate_reasoning(features, category, priority)
        
        return ClassificationResult(
            query_hash=features.query_hash,
            category=category,
            category_confidence=0.7,  # Lower confidence for rule-based
            optimization_priority=priority,
            priority_confidence=0.7,
            optimization_types=opt_types,
            reasoning=reasoning
        )
    
    def _suggest_optimization_types(
        self, 
        features: QueryFeatures, 
        category: QueryCategory
    ) -> List[str]:
        """Suggest optimization types based on features and category."""
        suggestions = []
        fd = features.feature_dict
        
        # Index-related
        if fd.get("where_condition_count", 0) > 0:
            suggestions.append("index_optimization")
        
        # Query rewrite
        if fd.get("has_select_star", 0):
            suggestions.append("select_column_specification")
        if fd.get("has_not_in", 0):
            suggestions.append("not_in_to_not_exists")
        if fd.get("has_leading_wildcard", 0):
            suggestions.append("like_pattern_optimization")
        if fd.get("has_or_condition", 0):
            suggestions.append("or_to_in_conversion")
        
        # Caching
        if category in [QueryCategory.REPORTING, QueryCategory.AGGREGATION]:
            suggestions.append("result_caching")
        
        # Join optimization
        if fd.get("join_count", 0) >= 2:
            suggestions.append("join_order_optimization")
        if fd.get("has_implicit_join", 0):
            suggestions.append("explicit_join_syntax")
        
        # Limiting
        if not fd.get("has_limit", 0) and category != QueryCategory.AGGREGATION:
            suggestions.append("add_limit_clause")
        
        return suggestions
    
    def _generate_reasoning(
        self,
        features: QueryFeatures,
        category: QueryCategory,
        priority: OptimizationPriority
    ) -> str:
        """Generate human-readable reasoning for the classification."""
        fd = features.feature_dict
        reasons = []
        
        # Category reasoning
        if category == QueryCategory.SIMPLE_LOOKUP:
            reasons.append("Simple single-table lookup query")
        elif category == QueryCategory.AGGREGATION:
            reasons.append(f"Aggregation query with {int(fd.get('group_by_count', 0))} GROUP BY columns")
        elif category == QueryCategory.JOIN_HEAVY:
            reasons.append(f"Complex query with {int(fd.get('join_count', 0))} joins")
        elif category == QueryCategory.ANALYTICAL:
            reasons.append("Analytical query with advanced SQL features")
        elif category == QueryCategory.REPORTING:
            reasons.append("Reporting query potentially returning large result sets")
        else:
            reasons.append("Transactional query pattern")
        
        # Priority reasoning
        complexity = fd.get("estimated_complexity_score", 0)
        reasons.append(f"Complexity score: {complexity:.1f}")
        
        # Specific issues
        if fd.get("has_select_star", 0):
            reasons.append("Uses SELECT * which can be optimized")
        if fd.get("has_leading_wildcard", 0):
            reasons.append("Leading wildcard in LIKE prevents index usage")
        if fd.get("has_function_in_where", 0):
            reasons.append("Function on column in WHERE may prevent index usage")
        if fd.get("has_not_in", 0):
            reasons.append("NOT IN subquery can be converted to NOT EXISTS")
        
        return ". ".join(reasons)
    
    def save(self, path: str):
        """Save the trained models to disk."""
        model_data = {
            "category_model": self.category_model,
            "priority_model": self.priority_model,
            "scaler": self.scaler,
            "category_encoder": self.category_encoder,
            "priority_encoder": self.priority_encoder,
            "model_version": self.model_version,
            "is_trained": self.is_trained
        }
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, path: str):
        """Load trained models from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.category_model = model_data["category_model"]
        self.priority_model = model_data["priority_model"]
        self.scaler = model_data["scaler"]
        self.category_encoder = model_data["category_encoder"]
        self.priority_encoder = model_data["priority_encoder"]
        self.model_version = model_data["model_version"]
        self.is_trained = model_data["is_trained"]
    
    def classify_batch(self, queries: List[str]) -> List[ClassificationResult]:
        """Classify multiple queries."""
        return [self.classify(q) for q in queries]
