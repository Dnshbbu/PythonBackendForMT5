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
        """Make predictions using the loaded model with improved feature handling"""
        if not all([self.model, self.feature_scaler, self.target_scaler]):
            raise ValueError("Pipeline not loaded. Call load_pipeline() first.")
        
        try:
            df = data.copy()
            
            # First, identify what kind of features we're dealing with
            base_features = [col for col in df.columns 
                            if not any(x in col for x in ['_lag_', 'ma_', 'momentum_', 'price_rel_'])]
            
            print(f"Base features available: {base_features}")
            
            # Create time features if needed
            if any(f in self.feature_columns for f in ['hour', 'day_of_week', 'day_of_month', 'month']):
                try:
                    df = FeatureProcessor.create_time_features(df)
                    print("Time features created successfully")
                except Exception as e:
                    print(f"Warning: Could not create time features: {str(e)}")
            
            # Create technical features if needed
            if any(f in self.feature_columns for f in ['ma_', 'momentum_', 'price_rel_']):
                if 'Price' in df.columns:
                    df = FeatureProcessor.create_technical_features(df)
                    print("Technical features created successfully")
                else:
                    print("Warning: 'Price' column not found, skipping technical features")
            
            # Create lag features for all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df = FeatureProcessor.create_lag_features(df, numeric_cols)
            print("Lag features created successfully")
            
            # Forward fill any NaN values
            df = df.ffill().bfill()
            
            # Check which required features are available
            available_features = set(df.columns)
            required_features = set(self.feature_columns)
            missing_features = required_features - available_features
            
            if missing_features:
                # Try to identify which types of features are missing
                missing_time = [f for f in missing_features if f in ['hour', 'day_of_week', 'day_of_month', 'month']]
                missing_tech = [f for f in missing_features if any(x in f for x in ['ma_', 'momentum_', 'price_rel_'])]
                missing_lag = [f for f in missing_features if '_lag_' in f]
                
                error_msg = "Missing features detected:\n"
                if missing_time:
                    error_msg += f"\nTime features: {missing_time}"
                if missing_tech:
                    error_msg += f"\nTechnical features: {missing_tech}"
                if missing_lag:
                    error_msg += f"\nLag features: {missing_lag}"
                    
                error_msg += "\n\nPlease ensure your input data contains:"
                error_msg += "\n- Date and Time columns for time features"
                error_msg += "\n- Price column for technical features"
                error_msg += "\n- Required base features for lag calculations"
                
                raise ValueError(error_msg)
            
            # Select and order features according to the model's requirements
            X = df[self.feature_columns]
            
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
            raise Exception(f"Error during prediction: {str(e)}")

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
    """Create and save pipeline from analyzer instance with additional metadata"""
    pipeline = ModelPipeline()
    
    # Identify base features (features without derived indicators)
    base_features = [col for col in feature_columns 
                    if not any(x in col for x in ['_lag_', 'ma_', 'momentum_', 'price_rel_'])]
    
    # Create additional metadata
    additional_metadata = {
        'training_shape': X.shape,
        'analyzer_params': {
            'n_splits': analyzer.n_splits,
            'sequence_length': analyzer.sequence_length
        },
        'base_features': base_features,  # Add base features to metadata
        'feature_types': {
            'base': base_features,
            'technical': [col for col in feature_columns if any(x in col for x in ['ma_', 'momentum_', 'price_rel_'])],
            'lagged': [col for col in feature_columns if '_lag_' in col],
            'time': ['hour', 'day_of_week', 'day_of_month', 'month']
        }
    }
    
    # Save pipeline
    pipeline.save_pipeline(
        model=analyzer.models['xgboost'],
        feature_scaler=analyzer.scaler_X,
        target_scaler=analyzer.scaler_y,
        feature_columns=feature_columns,
        base_path=base_path,
        additional_metadata=additional_metadata
    )
    
    return pipeline


# Add this to model_pipeline.py

class FeatureProcessor:
    """Class to handle feature creation consistently between training and prediction"""
    
    @staticmethod
    def create_time_features(df):
        """Create time-based features"""
        if 'DateTime' not in df.columns:
            if 'Date' in df.columns and 'Time' in df.columns:
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            else:
                raise ValueError("DateTime or Date/Time columns required")
        
        df['hour'] = df['DateTime'].dt.hour
        df['day_of_week'] = df['DateTime'].dt.dayofweek
        df['day_of_month'] = df['DateTime'].dt.day
        df['month'] = df['DateTime'].dt.month
        return df
    
    @staticmethod
    def create_technical_features(df, price_col='Price'):
        """Create technical indicators"""
        for window in [5, 10, 20]:
            df[f'ma_{window}'] = df[price_col].rolling(window=window).mean()
            df[f'ma_std_{window}'] = df[price_col].rolling(window=window).std()
            df[f'momentum_{window}'] = df[price_col].diff(window)
        
        # Relative price features
        df['price_rel_ma5'] = df[price_col] / df['ma_5']
        df['price_rel_ma10'] = df[price_col] / df['ma_10']
        return df
    
    @staticmethod
    def create_lag_features(df, columns, lags=[1, 2, 3]):
        """Create lagged features"""
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return df
    
    @staticmethod
    def process_features(df, base_features, create_time_features=True):
        """Process all features consistently"""
        df = df.copy()
        
        # Create time features
        if create_time_features:
            df = FeatureProcessor.create_time_features(df)
        
        # Create technical features if Price column exists
        if 'Price' in df.columns:
            df = FeatureProcessor.create_technical_features(df)
        
        # Create lag features for base features
        df = FeatureProcessor.create_lag_features(df, base_features)
        
        # Forward fill any NaN values
        df = df.ffill().bfill()
        
        return df