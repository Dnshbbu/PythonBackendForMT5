"""
model_manager.py - Handles saving, loading, and managing ML models
"""
import os
import joblib
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd

@dataclass
class ModelInfo:
    """Class for storing model metadata"""
    name: str
    model_type: str  # 'regression' or 'classification'
    algorithm: str
    features: List[str]
    target: str
    metrics: Dict[str, float]
    creation_date: str

class ModelManager:
    """Handles saving, loading, and managing machine learning models"""
    
    def __init__(self, base_path: str = "models"):
        """
        Initialize ModelManager
        
        Args:
            base_path: Directory where models will be stored
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def save_model(self, model: Any, scaler: Any, model_info: ModelInfo) -> str:
        """
        Save model, scaler, and metadata
        
        Args:
            model: Trained model object
            scaler: Fitted scaler object
            model_info: ModelInfo object containing metadata
            
        Returns:
            str: Path to saved model directory
        """
        # Create model directory with timestamp
        model_dir = os.path.join(
            self.base_path,
            f"{model_info.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            # Save model components
            joblib.dump(model, os.path.join(model_dir, "model.joblib"))
            joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
            joblib.dump(model_info, os.path.join(model_dir, "metadata.joblib"))
            
            # Verify files were saved
            for filename in ["model.joblib", "scaler.joblib", "metadata.joblib"]:
                if not os.path.exists(os.path.join(model_dir, filename)):
                    raise FileNotFoundError(f"Failed to save {filename}")
            
            return model_dir
            
        except Exception as e:
            # Clean up if save failed
            if os.path.exists(model_dir):
                import shutil
                shutil.rmtree(model_dir)
            raise e
    
    def load_model(self, model_name: str) -> Tuple[Any, Any, ModelInfo]:
        """
        Load a saved model and its components
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Tuple containing (model, scaler, model_info)
        """
        model_dir = os.path.join(self.base_path, model_name)
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        try:
            model = joblib.load(os.path.join(model_dir, "model.joblib"))
            scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
            model_info = joblib.load(os.path.join(model_dir, "metadata.joblib"))
            
            return model, scaler, model_info
            
        except Exception as e:
            raise Exception(f"Error loading model components: {str(e)}")
    
    def predict(self, model_name: str, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using a saved model
        
        Args:
            model_name: Name of the model to use
            data: DataFrame containing features
            
        Returns:
            numpy.ndarray: Model predictions
        """
        # Load model components
        model, scaler, model_info = self.load_model(model_name)
        
        # Validate features
        missing_features = set(model_info.features) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Prepare features and predict
        X = data[model_info.features].copy()
        X_scaled = scaler.transform(X)
        
        return model.predict(X_scaled)
    
    def list_models(self) -> List[ModelInfo]:
        """
        List all saved models
        
        Returns:
            List[ModelInfo]: List of model metadata
        """
        models = []
        
        if not os.path.exists(self.base_path):
            return models
            
        for model_dir in os.listdir(self.base_path):
            try:
                _, _, model_info = self.load_model(model_dir)
                models.append(model_info)
            except Exception:
                continue
                
        return models
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a saved model
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            bool: True if deletion was successful
        """
        model_dir = os.path.join(self.base_path, model_name)
        
        if not os.path.exists(model_dir):
            return False
            
        try:
            import shutil
            shutil.rmtree(model_dir)
            return True
        except Exception:
            return False

def create_model_info(
    name: str,
    model_type: str,
    algorithm: str,
    features: List[str],
    target: str,
    metrics: Dict[str, float]
) -> ModelInfo:
    """
    Helper function to create ModelInfo instances
    
    Args:
        name: Model name
        model_type: Type of model (regression/classification)
        algorithm: Algorithm name
        features: List of feature names
        target: Target variable name
        metrics: Dictionary of model metrics
        
    Returns:
        ModelInfo object
    """
    return ModelInfo(
        name=name,
        model_type=model_type,
        algorithm=algorithm,
        features=features,
        target=target,
        metrics=metrics,
        creation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )