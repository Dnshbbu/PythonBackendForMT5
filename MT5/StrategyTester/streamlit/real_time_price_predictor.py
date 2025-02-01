import os
import logging
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime
from model_predictor import ModelPredictor
import json
from feature_config import TECHNICAL_FEATURES, ENTRY_FEATURES

class RealTimePricePredictor:
    """
    Real-time price prediction system with consistent feature usage.
    """
    
    # def __init__(self, db_path: str, models_dir: str, batch_size: int = 10):
    #     """Initialize the predictor"""
    #     self.model_predictor = ModelPredictor(db_path, models_dir)
    #     self.batch_size = batch_size
    #     self.data_buffer = deque(maxlen=batch_size)
    #     self.last_prediction = None
    #     self.models_dir = models_dir
    #     self.setup_logging()
        
    #     # Load feature configuration
    #     self.selected_features = self.load_feature_config()
        
    #     if self.model_predictor.model:
    #         logging.info("Model loaded successfully")
    #         logging.info(f"Number of features: {len(self.selected_features)}")
    #         logging.info(f"Features: {self.selected_features}")

    def __init__(self, db_path: str, models_dir: str, training_manager=None):
        """Initialize the predictor"""
        self.model_predictor = ModelPredictor(db_path, models_dir)
        self.models_dir = models_dir
        self.setup_logging()
        
        # Add data buffer for maintaining price history
        self.data_buffer = deque(maxlen=2)  # Keep current and previous data points
        
        self.training_manager = training_manager
        self.selected_features = self.load_feature_config()
        
        if self.model_predictor.model:
            logging.info("Model loaded successfully")
            logging.info(f"Number of features: {len(self.selected_features)}")
            logging.info(f"Features: {self.selected_features}")

    def setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    # def load_feature_config(self) -> List[str]:
    #     """Load feature configuration saved during training"""
    #     config_path = os.path.join(self.models_dir, 'feature_config.json')
    #     try:
    #         with open(config_path, 'r') as f:
    #             config = json.load(f)
    #         return config['features']
    #     except Exception as e:
    #         logging.error(f"Error loading feature config: {e}")
    #         # Return default features if config not found
    #         return [
    #             'Factors_maScore', 'Factors_rsiScore', 'Factors_macdScore', 'Factors_stochScore',
    #             'Factors_bbScore', 'Factors_atrScore', 'Factors_sarScore', 'Factors_ichimokuScore',
    #             'Factors_adxScore', 'Factors_volumeScore', 'Factors_mfiScore', 'Factors_priceMAScore',
    #             'Factors_emaScore', 'Factors_emaCrossScore', 'Factors_cciScore',
    #             'EntryScore_AVWAP', 'EntryScore_EMA', 'EntryScore_SR'
    #         ]

    def load_feature_config(self) -> List[str]:
        """Load feature configuration saved during training"""
        try:
            # First try to load from feature_config.json
            config_path = os.path.join(self.models_dir, 'feature_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                return config['features']
                
            # If not found, try to get features from the latest model's feature importance file
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_feature_importance.json')]
            if model_files:
                latest_file = max(model_files, key=lambda x: os.path.getctime(os.path.join(self.models_dir, x)))
                with open(os.path.join(self.models_dir, latest_file), 'r') as f:
                    features = list(json.load(f).keys())
                
                # Save these features as feature_config.json for future use
                with open(config_path, 'w') as f:
                    json.dump({'features': features}, f, indent=4)
                
                return features
                
            # If no feature files found, use default features from xgboost_train_model.py
            # from xgboost_train_model import TECHNICAL_FEATURES, ENTRY_FEATURES
            default_features = TECHNICAL_FEATURES + ENTRY_FEATURES
            
            # Save default features as feature_config.json
            with open(config_path, 'w') as f:
                json.dump({'features': default_features}, f, indent=4)
                
            return default_features
            
        except Exception as e:
            logging.warning(f"Error loading feature config: {e}")
            # Return default feature list if everything else fails
            return [
                'Factors_maScore', 'Factors_rsiScore', 'Factors_macdScore', 'Factors_stochScore',
                'Factors_bbScore', 'Factors_atrScore', 'Factors_sarScore', 'Factors_ichimokuScore',
                'Factors_adxScore', 'Factors_volumeScore', 'Factors_mfiScore', 'Factors_priceMAScore',
                'Factors_emaScore', 'Factors_emaCrossScore', 'Factors_cciScore',
                'EntryScore_AVWAP', 'EntryScore_EMA', 'EntryScore_SR'
            ]

    def split_factor_string(self, column_name: str, factor_string: str) -> Dict[str, any]:
        """Split a factor string into individual components"""
        result = {}
        if not factor_string:
            return result
            
        parts = factor_string.split('|')
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                new_key = f"{column_name}_{key}"
                
                try:
                    result[new_key] = float(value)
                except ValueError:
                    if value.lower() == 'true':
                        result[new_key] = 1.0
                    elif value.lower() == 'false':
                        result[new_key] = 0.0
                    else:
                        result[new_key] = value
                        
        return result

    def process_raw_data(self, data_point: Dict) -> Dict:
        """Process raw data point into proper format for model"""
        try:
            processed_data = {}
            
            # Process basic numeric fields
            numeric_fields = ['Price', 'Score', 'ExitScore']
            for field in numeric_fields:
                try:
                    value = data_point.get(field, '0')
                    processed_data[field] = float(value) if value != '' else 0.0
                except (ValueError, TypeError):
                    processed_data[field] = 0.0
            
            # Process factor strings
            split_columns = ['Factors', 'ExitFactors', 'EntryScore']
            for column in split_columns:
                if column in data_point:
                    split_data = self.split_factor_string(column, data_point[column])
                    # Convert split data to numeric where possible
                    for key, value in split_data.items():
                        try:
                            if isinstance(value, str) and value.strip() != '':
                                split_data[key] = float(value)
                            elif value == '' or value is None:
                                split_data[key] = 0.0
                        except (ValueError, TypeError):
                            split_data[key] = 0.0
                    processed_data.update(split_data)
            
            return processed_data
            
        except Exception as e:
            logging.error(f"Error processing raw data: {e}")
            raise

    def prepare_features(self, data: Dict) -> pd.DataFrame:
        """Prepare features from single data point"""
        try:
            # Convert single data point to DataFrame
            df = pd.DataFrame([data])
            
            # Handle missing features
            missing_features = [f for f in self.selected_features if f not in df.columns]
            if missing_features:
                for feature in missing_features:
                    df[feature] = 0.0
            
            # Select only required features
            X = df[self.selected_features].copy()
            
            # Convert all columns to numeric, replacing errors with 0
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
            # Apply scaling if available
            if self.model_predictor.scaler:
                X = pd.DataFrame(
                    self.model_predictor.scaler.transform(X),
                    columns=self.selected_features,
                    index=X.index
                )
            
            logging.debug(f"Prepared features shape: {X.shape}")
            return X
            
        except Exception as e:
            logging.error(f"Error preparing features: {e}")
            raise

    def make_prediction(self, features: pd.DataFrame) -> Dict:
        """Make prediction for next price point"""
        try:
            # Make prediction for next price
            prediction = self.model_predictor.model.predict(features)
            
            # Get feature importance
            feature_importance = self.model_predictor.model.get_feature_importance()
            
            # Get top features
            top_features = dict(sorted(
                feature_importance.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:5])
            
            # Calculate confidence based on feature importance values
            confidence = min(sum(abs(v) for v in top_features.values()) / len(top_features), 1.0)
            
            prediction_value = float(prediction[0])
            
            # Get current price if available
            current_price = self.data_buffer[-1]['price'] if self.data_buffer else None
            
            result = {
                'prediction': prediction_value,
                'predicted_change': (prediction_value - current_price) if current_price else None,
                'confidence': float(confidence),
                'is_confident': bool(confidence >= 0.8),
                'top_features': top_features,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            raise

    def add_data_point(self, data_point: Dict) -> Optional[Dict]:
        """Process single data point and return prediction for next price"""
        try:
            # Check and reload model if needed
            if self.training_manager is not None:
                try:
                    training_status = self.training_manager.get_latest_training_status()
                    if training_status.get('status') == 'completed':
                        latest_model = training_status.get('model_path')
                        if latest_model and latest_model != getattr(self.model_predictor, 'current_model_name', None):
                            logging.info(f"New model available, reloading: {latest_model}")
                            self.model_predictor.load_latest_model()
                except Exception as e:
                    logging.warning(f"Error checking training status: {e}")

            # Process raw data
            processed_data = self.process_raw_data(data_point)
            current_price = float(data_point.get('Price', 0))
            
            # Add to data buffer
            self.data_buffer.append({
                'processed_data': processed_data,
                'price': current_price,
                'timestamp': data_point.get('Time')
            })
            
            # Only make prediction if we have previous data point
            if len(self.data_buffer) >= 1:
                # Prepare features from current data point
                features = self.prepare_features(processed_data)
                
                # Make prediction for next price
                prediction_result = self.make_prediction(features)
                
                # Add previous price info if available for comparison
                if len(self.data_buffer) == 2:
                    previous_price = self.data_buffer[0]['price']
                    price_change = current_price - previous_price
                    prediction_result.update({
                        'current_price': current_price,
                        'previous_price': previous_price,
                        'price_change': price_change,
                    })
                
                # Log prediction details
                logging.info(f"Time: {data_point.get('Time')}")
                logging.info(f"Current Price: {current_price}")
                logging.info(f"Predicted Next Price: {prediction_result['prediction']}")
                
                return prediction_result
            
            return None
            
        except Exception as e:
            logging.error(f"Error processing data point: {e}")
            raise

