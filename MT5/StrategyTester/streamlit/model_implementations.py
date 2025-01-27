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

class BaseTimeSeriesModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.feature_columns = None

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
        """Save model and metadata"""
        model_path = os.path.join(save_dir, f"{self.model_name}_{timestamp}.joblib")
        joblib.dump(self.model, model_path)
        
        if self.scaler:
            scaler_path = os.path.join(save_dir, f"{self.model_name}_{timestamp}_scaler.joblib")
            joblib.dump(self.scaler, scaler_path)

        # Save feature importance
        importance = self.get_feature_importance()
        importance_path = os.path.join(save_dir, f"{self.model_name}_{timestamp}_feature_importance.json")
        with open(importance_path, 'w') as f:
            json.dump(importance, f, indent=4)

        return model_path

    def load(self, model_path: str):
        """Load model and associated files"""
        self.model = joblib.load(model_path)
        base_path = model_path.rsplit('.', 1)[0]
        
        # Try to load scaler
        scaler_path = f"{base_path}_scaler.joblib"
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        # Try to load feature importance
        importance_path = f"{base_path}_feature_importance.json"
        if os.path.exists(importance_path):
            with open(importance_path, 'r') as f:
                self.feature_columns = list(json.load(f).keys())

class XGBoostTimeSeriesModel(BaseTimeSeriesModel):
    def __init__(self):
        super().__init__("xgboost")
        self.default_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'reg:squarederror'
        }

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[Any, Dict]:
        params = {**self.default_params, **kwargs}
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X, y)
        
        # Calculate metrics
        predictions = self.model.predict(X)
        metrics = {
            'training_score': self.model.score(X, y),
            'feature_importance': self.get_feature_importance()
        }
        
        return self.model, metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.model:
            return {}
            
        # Convert numpy importances to Python float
        importances = [float(imp) for imp in self.model.feature_importances_]
        columns = self.feature_columns
        
        if not columns and hasattr(self.model, 'feature_names_'):
            columns = self.model.feature_names_
            
        return dict(zip(columns, importances))

class DecisionTreeTimeSeriesModel(BaseTimeSeriesModel):
    def __init__(self):
        super().__init__("decision_tree")
        self.default_params = {
            'max_depth': 6,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[Any, Dict]:
        params = {**self.default_params, **kwargs}
        self.model = DecisionTreeRegressor(**params)
        self.model.fit(X, y)
        
        # Calculate metrics
        predictions = self.model.predict(X)
        metrics = {
            'training_score': self.model.score(X, y),
            'feature_importance': self.get_feature_importance()
        }
        
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
        """Register a new model type"""
        if not issubclass(model_class, BaseTimeSeriesModel):
            raise ValueError("Model must inherit from BaseTimeSeriesModel")
        cls._models[name.lower()] = model_class