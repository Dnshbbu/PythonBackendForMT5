from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

class BaseModel(ABC):
    """Base interface for all models"""
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[Any, Dict]:
        """Train the model and return (model, metrics)"""
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
        
    @abstractmethod
    def save(self, path: str) -> str:
        """Save model and return path"""
        pass
        
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from path"""
        pass
        
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        pass
        
    @property
    @abstractmethod
    def supports_incremental(self) -> bool:
        """Whether model supports incremental training"""
        pass
        
    def partial_fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """Incremental training if supported"""
        if not self.supports_incremental:
            raise NotImplementedError("Model does not support incremental training") 