from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any
import logging
from datetime import datetime

class IncrementalLearningMixin(ABC):
    """Mixin class to provide incremental learning capabilities"""
    
    @abstractmethod
    def supports_incremental_learning(self) -> bool:
        """Check if model supports incremental learning"""
        pass
        
    @abstractmethod
    def partial_fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """Incrementally train the model on new data"""
        pass
        
    @abstractmethod
    def needs_retraining(self, new_data_size: int) -> bool:
        """Check if model needs complete retraining"""
        pass

class BatchedRetrainingMixin:
    """Mixin class for models that don't support true incremental learning"""
    
    def __init__(self):
        self.training_history = []
        self.batch_size = 1000  # Default batch size
        self.retraining_threshold = 0.2  # Retrain if new data is 20% of original
        
    def needs_retraining(self, new_data_size: int) -> bool:
        """Determine if complete retraining is needed based on data size"""
        if not self.training_history:
            return True
            
        last_training = self.training_history[-1]
        original_size = last_training.get('data_size', 0)
        
        return new_data_size / original_size > self.retraining_threshold
        
    def record_training(self, metrics: Dict[str, Any], data_size: int):
        """Record training event and metrics"""
        training_event = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'data_size': data_size
        }
        self.training_history.append(training_event)
        
    def get_training_history(self) -> list:
        """Get complete training history"""
        return self.training_history

class ModelTrainingMetrics:
    """Class to track and compare model training metrics"""
    
    def __init__(self):
        self.metrics_history = []
        
    def add_metrics(self, metrics: Dict[str, Any], training_type: str):
        """Add new training metrics with metadata"""
        metrics_event = {
            'timestamp': datetime.now().isoformat(),
            'training_type': training_type,  # 'incremental' or 'full'
            'metrics': metrics
        }
        self.metrics_history.append(metrics_event)
        
    def compare_metrics(self, window: int = 5) -> Dict[str, Any]:
        """Compare recent metrics to historical performance"""
        if len(self.metrics_history) < 2:
            return {}
            
        recent = self.metrics_history[-window:]
        historical = self.metrics_history[:-window]
        
        if not historical:
            return {}
            
        comparison = {}
        metric_keys = recent[-1]['metrics'].keys()
        
        for key in metric_keys:
            if key in ['rmse', 'mae', 'r2']:
                recent_avg = np.mean([m['metrics'][key] for m in recent])
                hist_avg = np.mean([m['metrics'][key] for m in historical])
                comparison[key] = {
                    'recent_avg': recent_avg,
                    'historical_avg': hist_avg,
                    'change_pct': ((recent_avg - hist_avg) / hist_avg) * 100
                }
                
        return comparison

class ModelVersionManager:
    """Class to manage model versions and transitions"""
    
    def __init__(self):
        self.versions = []
        
    def add_version(self, model_path: str, metrics: Dict[str, Any], 
                   training_type: str, timestamp: Optional[str] = None):
        """Record new model version"""
        version = {
            'timestamp': timestamp or datetime.now().isoformat(),
            'model_path': model_path,
            'metrics': metrics,
            'training_type': training_type
        }
        self.versions.append(version)
        
    def get_latest_version(self) -> Optional[Dict]:
        """Get the most recent model version"""
        return self.versions[-1] if self.versions else None
        
    def should_switch_versions(self, new_metrics: Dict[str, Any], 
                             threshold: float = 0.1) -> bool:
        """Determine if we should switch to a new model version"""
        if not self.versions:
            return True
            
        latest = self.versions[-1]['metrics']
        metric_keys = ['rmse', 'r2']  # Key metrics to compare
        
        for key in metric_keys:
            if key in latest and key in new_metrics:
                if key == 'rmse':
                    # Lower RMSE is better
                    if new_metrics[key] < latest[key] * (1 - threshold):
                        return True
                elif key == 'r2':
                    # Higher R2 is better
                    if new_metrics[key] > latest[key] * (1 + threshold):
                        return True
                        
        return False