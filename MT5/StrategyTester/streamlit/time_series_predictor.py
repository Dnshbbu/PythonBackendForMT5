import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple, Any
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

def load_model(model_path: str) -> Any:
    """Load a trained time series model from the given path.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        The loaded model object
    """
    try:
        # Get model info to determine model type
        model_info = get_model_info(model_path)
        model_type = model_info.get('model_type', '').lower()
        
        # Load the appropriate model based on type
        if model_type == 'prophet':
            with open(os.path.join(model_path, 'model.json'), 'r') as f:
                model = Prophet.from_json(f.read())
        elif model_type in ['arima', 'sarima', 'auto arima']:
            model = joblib.load(os.path.join(model_path, 'model.joblib'))
        elif model_type == 'var':
            model = joblib.load(os.path.join(model_path, 'model.joblib'))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model
    
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {str(e)}")
        raise

def get_model_info(model_path: str) -> Dict[str, Any]:
    """Get information about a trained model from its metadata file.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Dictionary containing model information
    """
    try:
        # Look for metadata.json instead of model_info.json
        info_file = os.path.join(model_path, 'metadata.json')
        if not os.path.exists(info_file):
            raise FileNotFoundError(f"Model metadata file not found at {info_file}")
            
        with open(info_file, 'r') as f:
            model_info = json.load(f)
            
        return model_info
    except Exception as e:
        raise Exception(f"Error loading model metadata: {str(e)}")

def prepare_data_for_prediction(
    data: pd.DataFrame,
    model_info: Dict[str, Any]
) -> Union[pd.DataFrame, pd.Series]:
    """Prepare input data for prediction based on model requirements.
    
    Args:
        data: Input DataFrame containing features
        model_info: Model information dictionary
        
    Returns:
        Prepared data ready for prediction
    """
    try:
        model_type = model_info.get('model_type', '').lower()
        
        # Handle different model types
        if model_type == 'prophet':
            # Prophet requires 'ds' column for dates
            pred_data = pd.DataFrame()
            pred_data['ds'] = pd.to_datetime(data['DateTime'] if 'DateTime' in data.columns 
                                           else data['Date'] + ' ' + data['Time'])
            
            # Add regressor columns if any
            features = model_info.get('features', [])
            for feature in features:
                if feature in data.columns:
                    pred_data[feature] = data[feature]
            
            return pred_data
        
        elif model_type == 'var':
            # VAR models need all features in the correct order
            features = model_info.get('features', [])
            if not features:
                raise ValueError("No features specified in model info for VAR model")
            
            # Select and order columns according to training features
            pred_data = data[features].copy()
            
            # Handle any necessary preprocessing (e.g., scaling)
            if model_info.get('preprocessing', {}).get('scaling', False):
                scaler_path = os.path.join(os.path.dirname(model_info['model_path']), 'scaler.joblib')
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    pred_data = pd.DataFrame(
                        scaler.transform(pred_data),
                        columns=pred_data.columns,
                        index=pred_data.index
                    )
            
            return pred_data
        
        else:  # ARIMA, SARIMA, Auto ARIMA
            # These models typically need just the target series
            target_col = model_info.get('target_column')
            if not target_col:
                raise ValueError("No target column specified in model info")
            
            if target_col not in data.columns:
                raise ValueError(f"Target column {target_col} not found in input data")
            
            return data[target_col]
    
    except Exception as e:
        logging.error(f"Error preparing data for prediction: {str(e)}")
        raise

def make_predictions(
    model: Any,
    prepared_data: Union[pd.DataFrame, pd.Series]
) -> np.ndarray:
    """Generate predictions using the loaded model.
    
    Args:
        model: Loaded model object
        prepared_data: Prepared input data
        
    Returns:
        Array of predictions
    """
    try:
        if isinstance(model, Prophet):
            # Prophet prediction
            forecast = model.predict(prepared_data)
            return forecast['yhat'].values
        
        elif isinstance(model, VAR):
            # VAR prediction
            predictions = model.forecast(prepared_data.values, steps=1)
            return predictions
        
        elif isinstance(model, (SARIMAX)):
            # ARIMA/SARIMA prediction
            predictions = model.forecast(steps=len(prepared_data))
            return predictions
        
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
    
    except Exception as e:
        logging.error(f"Error making predictions: {str(e)}")
        raise

def save_predictions(
    predictions: np.ndarray,
    original_data: pd.DataFrame,
    output_path: str,
    model_info: Dict[str, Any]
) -> str:
    """Save predictions along with original data.
    
    Args:
        predictions: Array of predictions
        original_data: Original input DataFrame
        output_path: Path to save the predictions
        model_info: Model information dictionary
        
    Returns:
        Path to the saved predictions file
    """
    try:
        # Create results DataFrame
        results = original_data.copy()
        target_col = model_info.get('target_column', 'target')
        results[f'Predicted_{target_col}'] = predictions
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'predictions_{timestamp}.csv'
        filepath = os.path.join(output_path, filename)
        
        # Save to CSV
        results.to_csv(filepath, index=False)
        return filepath
    
    except Exception as e:
        logging.error(f"Error saving predictions: {str(e)}")
        raise 