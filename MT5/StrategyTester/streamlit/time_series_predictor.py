import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple
import os
import joblib
import json
import logging
from datetime import datetime
from prophet import Prophet
from statsmodels.tsa.vector_ar.var_model import VARResults
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from sklearn.preprocessing import StandardScaler

class TimeSeriesPredictor:
    def __init__(self, model_path: str):
        """Initialize predictor with a trained model
        
        Args:
            model_path: Path to the model directory containing model.pkl and metadata.json
        """
        self.model_path = model_path
        self.model = None
        self.metadata = None
        self.model_type = None
        self.target = None
        self.features = None
        self.n_lags = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and its metadata"""
        try:
            # Load model
            model_file = os.path.join(self.model_path, 'model.pkl')
            self.model = joblib.load(model_file)
            
            # Load metadata
            metadata_file = os.path.join(self.model_path, 'metadata.json')
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            
            # Extract key information
            self.model_type = self.metadata['model_type']
            self.target = self.metadata['target']
            self.features = self.metadata['features']
            self.n_lags = self.metadata['n_lags']
            
            logging.info(f"Loaded {self.model_type} model from {self.model_path}")
            logging.info(f"Target: {self.target}")
            logging.info(f"Features: {self.features}")
            logging.info(f"Number of lags: {self.n_lags}")
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for prediction
        
        Args:
            data: DataFrame containing required features
            
        Returns:
            Prepared DataFrame ready for prediction
        """
        df = data.copy()
        
        # Add lagged features if needed
        if self.n_lags > 0:
            for i in range(1, self.n_lags + 1):
                lag_col = f"{self.target}_lag_{i}"
                df[lag_col] = df[self.target].shift(i)
        
        # Drop rows with NaN from lagged features
        if self.n_lags > 0:
            df = df.iloc[self.n_lags:]
        
        return df
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Make predictions using the loaded model
        
        Args:
            data: DataFrame containing required features
            
        Returns:
            Tuple of (predictions, prediction_info)
        """
        try:
            prepared_data = self.prepare_data(data)
            
            if self.model_type == 'Prophet':
                # Prepare Prophet data
                prophet_df = pd.DataFrame({
                    'ds': prepared_data.index,
                    'y': prepared_data[self.target]
                })
                
                # Add regressors
                for feature in self.features:
                    if feature != self.target:
                        prophet_df[feature] = prepared_data[feature]
                
                # Make prediction
                forecast = self.model.predict(prophet_df)
                predictions = forecast['yhat'].values[-1:]  # Get only the last prediction
                prediction_info = {
                    'lower_bound': forecast['yhat_lower'].values[-1:],
                    'upper_bound': forecast['yhat_upper'].values[-1:],
                    'components': {
                        'trend': forecast['trend'].values[-1:],
                        'weekly': forecast['weekly'].values[-1:] if 'weekly' in forecast else None,
                        'yearly': forecast['yearly'].values[-1:] if 'yearly' in forecast else None
                    }
                }
            
            elif self.model_type == 'VAR':
                # Get feature values for the last observation
                all_features = [self.target] + [f for f in self.features if f != self.target]
                feature_data = prepared_data[all_features].values
                
                # Apply differencing based on metadata
                diff_orders = self.metadata.get('diff_orders', {})
                for i, feature in enumerate(all_features):
                    if diff_orders.get(feature, 0) > 0:
                        for _ in range(diff_orders[feature]):
                            feature_data[:, i] = np.diff(feature_data[:, i], prepend=feature_data[0, i])
                
                # Scale the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(feature_data)
                
                # Get the last observation for prediction
                last_obs = scaled_data[-1:]
                
                # Make prediction
                predictions = self.model.forecast(last_obs, steps=1)
                
                # Inverse transform the predictions
                predictions = scaler.inverse_transform(predictions)
                
                # Inverse differencing if needed
                for i, feature in enumerate(all_features):
                    if diff_orders.get(feature, 0) > 0:
                        for _ in range(diff_orders[feature]):
                            predictions[:, i] = np.cumsum(predictions[:, i]) + feature_data[0, i]
                
                # Extract the target variable prediction (first column)
                predictions = predictions[:, 0]
                
                # Get confidence intervals (if available)
                try:
                    forecast_obj = self.model.get_forecast(last_obs)
                    confidence_intervals = forecast_obj.conf_int()
                    prediction_info = {
                        'lower_bound': scaler.inverse_transform(confidence_intervals[:, 0])[:, 0],
                        'upper_bound': scaler.inverse_transform(confidence_intervals[:, 1])[:, 0],
                        'feature_contributions': None
                    }
                except:
                    prediction_info = {
                        'lower_bound': None,
                        'upper_bound': None,
                        'feature_contributions': None
                    }
            
            elif self.model_type in ['ARIMA', 'SARIMA', 'Auto ARIMA']:
                # Get feature values
                feature_data = prepared_data[self.features].values
                
                # Make prediction
                forecast = self.model.forecast(steps=1)
                predictions = np.array([forecast])  # Convert to numpy array and ensure it's 1D
                
                # Get confidence intervals
                forecast_obj = self.model.get_forecast(steps=1)
                confidence_intervals = forecast_obj.conf_int()
                
                prediction_info = {
                    'lower_bound': np.array([confidence_intervals.iloc[0, 0]]),  # Get first value for lower bound
                    'upper_bound': np.array([confidence_intervals.iloc[0, 1]])   # Get first value for upper bound
                }
            
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            return predictions, prediction_info
            
        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_type': self.model_type,
            'target': self.target,
            'features': self.features,
            'n_lags': self.n_lags,
            'metrics': self.metadata.get('metrics', {}),
            'training_date': self.metadata.get('training_date', None)
        }
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate if the data has all required features
        
        Args:
            data: DataFrame to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        missing_features = []
        for feature in self.features:
            if feature not in data.columns and not feature.startswith(f"{self.target}_lag_"):
                missing_features.append(feature)
        
        if missing_features:
            logging.error(f"Missing features: {missing_features}")
            return False
        
        return True 