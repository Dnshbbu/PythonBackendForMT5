"""
model_pipeline.py - Enhanced pipeline for model management and predictions with backward compatibility
"""
import os
import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.preprocessing import RobustScaler

class ModelPipeline:
    """Enhanced pipeline for managing ML models and making predictions"""
    
    def __init__(self):
        self.model: Optional[BaseEstimator] = None
        self.feature_scaler: Optional[RobustScaler] = None
        self.target_scaler: Optional[RobustScaler] = None
        self.feature_columns: Optional[List[str]] = None
        self.metadata: Dict[str, Any] = {}
        
    def save_pipeline(self, 
                     model: BaseEstimator, 
                     feature_scaler: RobustScaler, 
                     target_scaler: RobustScaler, 
                     feature_columns: List[str], 
                     base_path: str = 'models',
                     additional_metadata: Dict[str, Any] = None) -> str:
        """
        Save all components needed for prediction
        
        Args:
            model: Trained model
            feature_scaler: Fitted feature scaler
            target_scaler: Fitted target scaler
            feature_columns: List of feature column names
            base_path: Directory to save model files
            additional_metadata: Optional additional metadata to save
            
        Returns:
            str: Path to saved model directory
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(base_path, exist_ok=True)
            
            # Update instance attributes
            self.model = model
            self.feature_scaler = feature_scaler
            self.target_scaler = target_scaler
            self.feature_columns = feature_columns
            
            # Create metadata
            self.metadata = {
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'feature_columns': feature_columns,
                'model_type': type(model).__name__,
                'framework_versions': {
                    'numpy': np.__version__,
                    'pandas': pd.__version__,
                    'joblib': joblib.__version__
                }
            }
            
            # Add additional metadata if provided
            if additional_metadata:
                self.metadata.update(additional_metadata)
            
            # Create pipeline components dictionary
            pipeline_components = {
                'model': model,
                'feature_scaler': feature_scaler,
                'target_scaler': target_scaler,
                'feature_columns': feature_columns,  # Include feature columns in components
                'metadata': self.metadata  # Include metadata in components
            }
            
            # Save all components in a single file
            joblib.dump(pipeline_components, os.path.join(base_path, 'model_pipeline.joblib'))
            
            # Also save metadata separately for easier access
            with open(os.path.join(base_path, 'metadata.json'), 'w') as f:
                json.dump(self.metadata, f, indent=4)
            
            return base_path
            
        except Exception as e:
            raise Exception(f"Error saving pipeline: {str(e)}")
    
    def load_pipeline(self, base_path: str = 'models') -> None:
        """
        Load saved pipeline components with backward compatibility
        
        Args:
            base_path: Directory containing saved model files
        """
        try:
            # Load pipeline components
            pipeline_path = os.path.join(base_path, 'model_pipeline.joblib')
            if not os.path.exists(pipeline_path):
                raise FileNotFoundError(f"Pipeline file not found at {pipeline_path}")
            
            components = joblib.load(pipeline_path)
            
            # Handle different component storage formats
            if isinstance(components, dict):
                # New format with all components in one dictionary
                self.model = components['model']
                self.feature_scaler = components['feature_scaler']
                self.target_scaler = components['target_scaler']
                
                # Try to get feature columns from components first
                if 'feature_columns' in components:
                    self.feature_columns = components['feature_columns']
                else:
                    # Try to load from separate file (old format)
                    try:
                        with open(os.path.join(base_path, 'feature_columns.json'), 'r') as f:
                            self.feature_columns = json.load(f)
                    except FileNotFoundError:
                        print("Warning: No feature columns found. Some functionality may be limited.")
                        self.feature_columns = []
                
                # Try to get metadata from components first
                if 'metadata' in components:
                    self.metadata = components['metadata']
                else:
                    # Try to load from separate file
                    try:
                        with open(os.path.join(base_path, 'metadata.json'), 'r') as f:
                            self.metadata = json.load(f)
                    except FileNotFoundError:
                        print("Warning: No metadata found. Using basic metadata.")
                        self.metadata = {
                            'feature_columns': self.feature_columns,
                            'model_type': type(self.model).__name__,
                            'creation_date': 'Unknown'
                        }
            
        except Exception as e:
            raise Exception(f"Error loading pipeline: {str(e)}")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the loaded model
        
        Args:
            data: DataFrame containing features
            
        Returns:
            numpy.ndarray: Model predictions
        """
        if not all([self.model, self.feature_scaler, self.target_scaler]):
            raise ValueError("Pipeline not loaded. Call load_pipeline() first.")
        
        try:
            # If feature columns are available, validate them
            if self.feature_columns:
                missing_features = set(self.feature_columns) - set(data.columns)
                if missing_features:
                    raise ValueError(f"Missing required features: {missing_features}")
                X = data[self.feature_columns].copy()
            else:
                # If no feature columns are specified, use all available columns
                print("Warning: No feature columns specified. Using all available columns.")
                X = data.copy()
            
            # Scale features
            X_scaled = self.feature_scaler.transform(X)
            
            # Make predictions
            predictions_scaled = self.model.predict(X_scaled)
            
            # Inverse transform predictions
            predictions = self.target_scaler.inverse_transform(
                predictions_scaled.reshape(-1, 1)
            ).ravel()
            
            return predictions
            
        except Exception as e:
            raise Exception(f"Error making predictions: {str(e)}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get pipeline metadata"""
        return self.metadata

    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns"""
        return self.feature_columns

def create_pipeline_from_analyzer(analyzer: Any, 
                                X: np.ndarray, 
                                feature_columns: List[str], 
                                base_path: str = 'models') -> ModelPipeline:
    """
    Create and save pipeline from analyzer instance
    
    Args:
        analyzer: Trained analyzer instance
        X: Feature matrix used for training
        feature_columns: List of feature column names
        base_path: Directory to save model files
        
    Returns:
        ModelPipeline: Configured pipeline instance
    """
    pipeline = ModelPipeline()
    
    # Create additional metadata
    additional_metadata = {
        'training_shape': X.shape,
        'analyzer_params': {
            'n_splits': analyzer.n_splits,
            'sequence_length': analyzer.sequence_length
        }
    }
    
    # Save pipeline
    pipeline.save_pipeline(
        model=analyzer.models['xgboost'],  # Using XGBoost as default
        feature_scaler=analyzer.scaler_X,
        target_scaler=analyzer.scaler_y,
        feature_columns=feature_columns,
        base_path=base_path,
        additional_metadata=additional_metadata
    )
    
    return pipeline