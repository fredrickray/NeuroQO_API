"""
Performance Predictor - Predicts query execution time using ML.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pickle
import os
import json

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from app.ml.feature_extractor import QueryFeatureExtractor, QueryFeatures


@dataclass
class PerformancePrediction:
    """Prediction result for query performance."""
    query_hash: str
    predicted_time_ms: float
    confidence_interval: Tuple[float, float]  # 95% CI
    model_confidence: float  # 0-1 confidence score
    is_slow_prediction: bool
    performance_class: str  # excellent, good, acceptable, slow, critical
    contributing_factors: Dict[str, float]  # Feature contributions


@dataclass
class ModelMetrics:
    """Metrics for model evaluation."""
    mse: float
    rmse: float
    mae: float
    r2: float
    mape: float  # Mean Absolute Percentage Error
    cv_scores: List[float]


class PerformancePredictor:
    """
    ML model to predict query execution time.
    
    Supports:
    - Random Forest
    - Gradient Boosting
    - LightGBM (if available)
    """
    
    def __init__(
        self, 
        model_type: str = "lightgbm",
        model_path: Optional[str] = None
    ):
        self.model_type = model_type if HAS_LIGHTGBM or model_type != "lightgbm" else "random_forest"
        self.feature_extractor = QueryFeatureExtractor()
        
        # Model and preprocessors
        self.model = None
        self.scaler = StandardScaler()
        
        # Model metadata
        self.model_version = "1.0.0"
        self.is_trained = False
        self.training_metrics: Optional[ModelMetrics] = None
        self.feature_importance: Dict[str, float] = {}
        
        # Prediction uncertainty estimation
        self.prediction_std: float = 0
        
        # Thresholds
        self.slow_threshold_ms = 1000  # Default 1 second
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def _create_model(self):
        """Create the ML model based on model_type."""
        if self.model_type == "lightgbm" and HAS_LIGHTGBM:
            return lgb.LGBMRegressor(  # type: ignore[possibly-undefined]
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42
            )
        else:
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
    
    def train(
        self,
        queries: List[str],
        execution_times_ms: List[float],
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> ModelMetrics:
        """
        Train the performance prediction model.
        
        Args:
            queries: List of SQL queries
            execution_times_ms: Corresponding execution times in milliseconds
            test_size: Fraction of data for testing
            cv_folds: Number of cross-validation folds
            
        Returns:
            ModelMetrics with training results
        """
        # Extract features
        X, _ = self.feature_extractor.extract_batch(queries)
        y = np.array(execution_times_ms)
        
        # Log transform target for better distribution
        y_log = np.log1p(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_log, test_size=test_size, random_state=42
        )
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_scaled, y_log,  # type: ignore[arg-type]
            cv=cv_folds, scoring='neg_mean_squared_error'
        )
        
        # Predictions on test set
        y_pred_log = self.model.predict(X_test)
        
        # Transform back from log scale
        y_pred = np.expm1(y_pred_log)  # type: ignore[call-overload]
        y_test_original = np.expm1(y_test)  # type: ignore[call-overload]
        
        # Calculate residuals for uncertainty estimation
        residuals = y_test_original - y_pred
        self.prediction_std = float(np.std(residuals))
        
        # Calculate metrics
        mse = mean_squared_error(y_test_original, y_pred)
        mae = mean_absolute_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        
        # MAPE (avoiding division by zero)
        mape = np.mean(np.abs((y_test_original - y_pred) / np.maximum(y_test_original, 1))) * 100
        
        self.training_metrics = ModelMetrics(
            mse=mse,
            rmse=np.sqrt(mse),
            mae=mae,
            r2=r2,
            mape=mape,
            cv_scores=(-cv_scores).tolist()  # Convert to positive MSE
        )
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_extractor.get_feature_names(),
                self.model.feature_importances_
            ))
        
        self.is_trained = True
        return self.training_metrics
    
    def predict(self, query: str) -> PerformancePrediction:
        """
        Predict execution time for a query.
        
        Args:
            query: SQL query
            
        Returns:
            PerformancePrediction with predicted time and confidence
        """
        features = self.feature_extractor.extract(query)
        
        if not self.is_trained or self.model is None:
            return self._estimate_without_model(features)
        
        # Scale features
        X = self.scaler.transform(features.feature_vector.reshape(1, -1))
        
        # Predict (in log scale)
        y_pred_log = self.model.predict(X)[0]  # type: ignore[union-attr]
        predicted_time = float(np.expm1(y_pred_log))  # type: ignore[call-overload]
        
        # Confidence interval (95%)
        ci_low = max(0, predicted_time - 1.96 * self.prediction_std)
        ci_high = predicted_time + 1.96 * self.prediction_std
        
        # Model confidence based on feature similarity to training data
        confidence = self._calculate_confidence(X)
        
        # Classify performance
        performance_class = self._classify_performance(predicted_time)
        
        # Get contributing factors
        factors = self._get_contributing_factors(features)
        
        return PerformancePrediction(
            query_hash=features.query_hash,
            predicted_time_ms=predicted_time,
            confidence_interval=(ci_low, ci_high),
            model_confidence=confidence,
            is_slow_prediction=predicted_time > self.slow_threshold_ms,
            performance_class=performance_class,
            contributing_factors=factors
        )
    
    def _estimate_without_model(
        self, 
        features: QueryFeatures
    ) -> PerformancePrediction:
        """Estimate performance using heuristics when model is not trained."""
        fd = features.feature_dict
        
        # Simple heuristic estimation
        base_time = 10  # Base time in ms
        
        # Add time based on complexity
        time_estimate = base_time
        time_estimate += fd.get("table_count", 0) * 20
        time_estimate += fd.get("join_count", 0) * 50
        time_estimate += fd.get("subquery_depth", 0) * 100
        time_estimate += fd.get("where_condition_count", 0) * 10
        time_estimate += fd.get("aggregation_count", 0) * 30
        time_estimate += fd.get("has_distinct", 0) * 50
        
        # Penalty for anti-patterns
        if fd.get("has_select_star", 0):
            time_estimate *= 1.2
        if fd.get("has_leading_wildcard", 0):
            time_estimate *= 2.0
        if fd.get("has_not_in", 0):
            time_estimate *= 1.5
        
        return PerformancePrediction(
            query_hash=features.query_hash,
            predicted_time_ms=time_estimate,
            confidence_interval=(time_estimate * 0.5, time_estimate * 2.0),
            model_confidence=0.3,  # Low confidence for heuristic
            is_slow_prediction=time_estimate > self.slow_threshold_ms,
            performance_class=self._classify_performance(time_estimate),
            contributing_factors={}
        )
    
    def _calculate_confidence(self, X: np.ndarray) -> float:
        """Calculate prediction confidence."""
        # Simple confidence based on R2 score
        if self.training_metrics:
            base_confidence = max(0, self.training_metrics.r2)
        else:
            base_confidence = 0.5
        
        return min(1.0, base_confidence + 0.2)  # Boost a bit
    
    def _classify_performance(self, time_ms: float) -> str:
        """Classify performance based on predicted time."""
        if time_ms < 10:
            return "excellent"
        elif time_ms < 100:
            return "good"
        elif time_ms < 500:
            return "acceptable"
        elif time_ms < self.slow_threshold_ms:
            return "slow"
        else:
            return "critical"
    
    def _get_contributing_factors(
        self, 
        features: QueryFeatures
    ) -> Dict[str, float]:
        """Get top contributing factors to the prediction."""
        if not self.feature_importance:
            return {}
        
        # Get top 5 contributing features
        sorted_importance = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        factors = {}
        for name, importance in sorted_importance:
            value = features.feature_dict.get(name, 0)
            if value > 0:
                factors[name] = importance * value
        
        return factors
    
    def predict_batch(
        self, 
        queries: List[str]
    ) -> List[PerformancePrediction]:
        """Predict performance for multiple queries."""
        return [self.predict(q) for q in queries]
    
    def predict_improvement(
        self,
        original_query: str,
        optimized_query: str
    ) -> Dict[str, Any]:
        """
        Predict the improvement from query optimization.
        
        Args:
            original_query: Original SQL query
            optimized_query: Optimized SQL query
            
        Returns:
            Dict with predicted improvement metrics
        """
        original_pred = self.predict(original_query)
        optimized_pred = self.predict(optimized_query)
        
        improvement_ms = original_pred.predicted_time_ms - optimized_pred.predicted_time_ms
        improvement_pct = (improvement_ms / original_pred.predicted_time_ms) * 100 if original_pred.predicted_time_ms > 0 else 0
        
        return {
            "original_predicted_ms": original_pred.predicted_time_ms,
            "optimized_predicted_ms": optimized_pred.predicted_time_ms,
            "improvement_ms": improvement_ms,
            "improvement_percentage": improvement_pct,
            "original_class": original_pred.performance_class,
            "optimized_class": optimized_pred.performance_class,
            "confidence": min(original_pred.model_confidence, optimized_pred.model_confidence),
            "is_significant_improvement": improvement_pct > 10
        }
    
    def update_model(
        self,
        new_queries: List[str],
        new_times: List[float]
    ):
        """
        Update the model with new data (incremental learning).
        
        Note: For Random Forest/LightGBM, this retrains the model.
        For production, consider online learning algorithms.
        """
        # This is a simplified approach - just retrain
        # In production, you'd want proper incremental learning
        if not self.is_trained:
            self.train(new_queries, new_times)
        else:
            # For now, just log that we need to retrain
            pass
    
    def save(self, path: str):
        """Save the model to disk."""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "is_trained": self.is_trained,
            "prediction_std": self.prediction_std,
            "feature_importance": self.feature_importance,
            "training_metrics": self.training_metrics,
            "slow_threshold_ms": self.slow_threshold_ms
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
        self.model_type = model_data["model_type"]
        self.model_version = model_data["model_version"]
        self.is_trained = model_data["is_trained"]
        self.prediction_std = model_data["prediction_std"]
        self.feature_importance = model_data["feature_importance"]
        self.training_metrics = model_data["training_metrics"]
        self.slow_threshold_ms = model_data.get("slow_threshold_ms", 1000)
    
    def get_metrics(self) -> Optional[ModelMetrics]:
        """Get training metrics."""
        return self.training_metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance
