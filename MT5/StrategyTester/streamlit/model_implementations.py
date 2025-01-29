from abc import ABC, abstractmethod
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from incremental_learning import (
    IncrementalLearningMixin, 
    BatchedRetrainingMixin,
    ModelTrainingMetrics,
    ModelVersionManager
)

class BaseTimeSeriesModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.metrics_tracker = ModelTrainingMetrics()
        self.version_manager = ModelVersionManager()

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[Any, Dict]:
        """Train the model and return metrics"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        pass

    def save(self, save_dir: str, timestamp: str) -> str:
        """Save model and metadata with proper type conversion"""
        try:
            model_path = os.path.join(save_dir, f"{self.model_name}_{timestamp}.joblib")
            joblib.dump(self.model, model_path)
            
            if self.scaler:
                scaler_path = os.path.join(save_dir, f"{self.model_name}_{timestamp}_scaler.joblib")
                joblib.dump(self.scaler, scaler_path)

            # Get and convert feature importance
            importance = self.get_feature_importance()
            # Ensure all values are Python native types
            converted_importance = {
                str(k): float(v) if hasattr(v, 'dtype') else float(v)
                for k, v in importance.items()
            }
            
            importance_path = os.path.join(save_dir, f"{self.model_name}_{timestamp}_feature_importance.json")
            with open(importance_path, 'w') as f:
                json.dump(converted_importance, f, indent=4)

            return model_path
            
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise

    def load(self, model_path: str):
        """Load model and associated files"""
        self.model = joblib.load(model_path)
        base_path = model_path.rsplit('.', 1)[0]
        
        scaler_path = f"{base_path}_scaler.joblib"
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        importance_path = f"{base_path}_feature_importance.json"
        if os.path.exists(importance_path):
            with open(importance_path, 'r') as f:
                self.feature_columns = list(json.load(f).keys())



class XGBoostTimeSeriesModel(BaseTimeSeriesModel, IncrementalLearningMixin):
    def __init__(self):
        super().__init__("xgboost")
        self.default_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'reg:squarederror'
        }

    def supports_incremental_learning(self) -> bool:
        return True

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[Any, Dict]:
        try:
            params = {**self.default_params, **kwargs}
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(X, y)
            
            # Get predictions
            predictions = self.model.predict(X)
            score = float(self.model.score(X, y))  # Convert score to Python float
            
            # Get feature importances with explicit conversion to Python types
            importance_dict = {}
            importances = self.model.feature_importances_
            feature_names = self.feature_columns or X.columns.tolist()
            
            for feat, imp in zip(feature_names, importances):
                importance_dict[str(feat)] = float(imp)
            
            metrics = {
                'training_score': score,
                'feature_importance': importance_dict,
                'n_features': len(feature_names),
                'n_samples': len(X)
            }
            
            self.metrics_tracker.add_metrics(metrics, 'full')
            return self.model, metrics
            
        except Exception as e:
            logging.error(f"Error in XGBoost training: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.model:
            return {}
        
        importances = {}
        feature_names = self.feature_columns or []
        for feat, imp in zip(feature_names, self.model.feature_importances_):
            importances[str(feat)] = float(imp)
        return importances
    

    def needs_retraining(self, new_data_size: int) -> bool:
        """Determine if XGBoost model needs retraining"""
        if not self.model:
            logging.info("No existing model found, needs full training")
            return True
            
        # Check if new data would increase trees beyond limit
        current_trees = self.model.n_estimators
        max_trees = 2000  # Maximum number of trees to maintain model efficiency
        
        # Calculate estimated new trees needed
        estimated_new_trees = int(new_data_size / 100)  # One tree per 100 samples
        total_trees = current_trees + estimated_new_trees
        
        if total_trees > max_trees:
            logging.info(f"Would exceed max trees ({max_trees}), needs full retraining")
            return True
            
        # Check if new data is too large compared to original training set
        if hasattr(self, 'training_history') and self.training_history:
            last_training = self.training_history[-1]
            original_size = last_training.get('data_size', 0)
            if original_size > 0 and new_data_size / original_size > 0.5:
                logging.info(f"New data too large compared to original training set")
                return True
                
        return False

    def partial_fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """Incrementally train XGBoost model"""
        if not self.model:
            _, metrics = self.train(X, y, **kwargs)
            return metrics
            
        try:
            n_new_trees = kwargs.get('n_estimators', 10)
            new_model = xgb.XGBRegressor(
                max_depth=self.model.max_depth,
                learning_rate=self.model.learning_rate,
                n_estimators=n_new_trees,
                objective=self.model.objective
            )
            
            new_model.fit(X, y)
            
            self.model.n_estimators += n_new_trees
            self.model._Booster = xgb.Booster({
                'nthread': self.model.n_jobs
            })
            self.model._Booster.feature_names = [str(f) for f in self.feature_columns]
            
            # Update metrics
            score = float(self.model.score(X, y))
            importance_dict = self.get_feature_importance()
            
            metrics = {
                'training_score': score,
                'feature_importance': importance_dict,
                'n_trees_added': n_new_trees,
                'n_samples': len(X)
            }
            
            self.metrics_tracker.add_metrics(metrics, 'incremental')
            return metrics
            
        except Exception as e:
            logging.error(f"Error in partial_fit: {e}")
            raise


class DecisionTreeTimeSeriesModel(BaseTimeSeriesModel, BatchedRetrainingMixin, IncrementalLearningMixin):
    def __init__(self):
        super().__init__("decision_tree")
        BatchedRetrainingMixin.__init__(self)
        self.default_params = {
            'max_depth': 6,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }

    def supports_incremental_learning(self) -> bool:
        return False  # Decision Trees don't support true incremental learning

    def partial_fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """Not implemented for Decision Trees"""
        raise NotImplementedError("Decision Trees do not support incremental learning")

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[Any, Dict]:
        params = {**self.default_params, **kwargs}
        self.model = DecisionTreeRegressor(**params)
        self.model.fit(X, y)
        
        predictions = self.model.predict(X)
        metrics = {
            'training_score': self.model.score(X, y),
            'feature_importance': self.get_feature_importance()
        }
        
        self.metrics_tracker.add_metrics(metrics, 'full')
        self.record_training(metrics, len(X))
        return self.model, metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.model:
            return {}
        return dict(zip(self.feature_columns or X.columns, 
                       self.model.feature_importances_))

class ModelFactory:
    _models = {
        'xgboost': XGBoostTimeSeriesModel,
        'decision_tree': DecisionTreeTimeSeriesModel
    }

    @classmethod
    def get_model(cls, model_type: str) -> BaseTimeSeriesModel:
        model_class = cls._models.get(model_type.lower())
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")
        return model_class()

    @classmethod
    def register_model(cls, name: str, model_class: type):
        if not issubclass(model_class, BaseTimeSeriesModel):
            raise ValueError("Model must inherit from BaseTimeSeriesModel")
        cls._models[name.lower()] = model_class