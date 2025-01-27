import os
import logging
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime
from model_predictor import ModelPredictor
import json

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

    def __init__(self, db_path: str, models_dir: str, batch_size: int = 10, training_manager=None):
            """Initialize the predictor"""
            self.model_predictor = ModelPredictor(db_path, models_dir)
            self.batch_size = batch_size
            self.data_buffer = deque(maxlen=batch_size)
            self.last_prediction = None
            self.models_dir = models_dir
            self.setup_logging()
            
            # Store training manager reference
            self.training_manager = training_manager
            
            # Load feature configuration
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

    def load_feature_config(self) -> List[str]:
        """Load feature configuration saved during training"""
        config_path = os.path.join(self.models_dir, 'feature_config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config['features']
        except Exception as e:
            logging.error(f"Error loading feature config: {e}")
            # Return default features if config not found
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

    def prepare_batch_features(self, batch_data: List[Dict]) -> pd.DataFrame:
        """Prepare features from batch data"""
        try:
            # Convert batch to DataFrame
            df = pd.DataFrame(batch_data)
            
            # Log initial data shape
            logging.info(f"Initial batch data shape: {df.shape}")
            
            # Handle missing features
            missing_features = [f for f in self.selected_features if f not in df.columns]
            if missing_features:
                missing_df = pd.DataFrame(0.0, index=df.index, columns=missing_features)
                df = pd.concat([df, missing_df], axis=1)
            
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
            
            logging.info(f"Final feature matrix shape: {X.shape}")
            return X
            
        except Exception as e:
            logging.error(f"Error preparing batch features: {e}")
            raise

    # def add_data_point(self, data_point: Dict) -> Optional[Dict]:
    #     """Add a new data point and make prediction if batch is full"""
    #     try:
    #         # Process raw data
    #         processed_data = self.process_raw_data(data_point)
    #         current_price = float(data_point.get('Price', 0))
            
    #         # Add to buffer
    #         self.data_buffer.append(processed_data)
            
    #         # Log buffer status
    #         logging.info(f"Buffer size: {len(self.data_buffer)}/{self.batch_size}")
            
    #         # Make prediction if buffer is full
    #         if len(self.data_buffer) == self.batch_size:
    #             prediction_result = self.make_prediction()
                
    #             # Log prediction change
    #             if self.last_prediction is not None:
    #                 # Get previous price from the second-to-last item in buffer
    #                 previous_data = list(self.data_buffer)[-2]
    #                 previous_price = float(previous_data.get('Price', 0))
    #                 price_change = current_price - previous_price
                    
    #                 pred_change = prediction_result['prediction'] - self.last_prediction
    #                 logging.info(f"Price change: {price_change:.4f}")
    #                 logging.info(f"Prediction change: {pred_change:.4f}")
                
    #             self.last_prediction = prediction_result['prediction']
    #             return prediction_result
            
    #         return None
            
    #     except Exception as e:
    #         logging.error(f"Error adding data point: {e}")
    #         raise

    def add_data_point(self, data_point: Dict) -> Optional[Dict]:
        """Add a new data point and make prediction if batch is full"""
        try:
            # Process raw data
            processed_data = self.process_raw_data(data_point)
            current_price = float(data_point.get('Price', 0))
            
            # Add to buffer
            self.data_buffer.append(processed_data)
            
            # Log buffer status
            logging.info(f"Buffer size: {len(self.data_buffer)}/{self.batch_size}")
            
            # Make prediction if buffer is full
            if len(self.data_buffer) == self.batch_size:
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

                prediction_result = self.make_prediction()
                
                # Log prediction change
                if self.last_prediction is not None:
                    # Get previous price from the second-to-last item in buffer
                    previous_data = list(self.data_buffer)[-2]
                    previous_price = float(previous_data.get('Price', 0))
                    price_change = current_price - previous_price
                    
                    pred_change = prediction_result['prediction'] - self.last_prediction
                    logging.info(f"Price change: {price_change:.4f}")
                    logging.info(f"Prediction change: {pred_change:.4f}")
                
                self.last_prediction = prediction_result['prediction']
                return prediction_result
            
            return None
            
        except Exception as e:
            logging.error(f"Error adding data point: {e}")
            raise

    def make_prediction(self) -> Dict:
        """Make prediction using current batch of data"""
        try:
            # Prepare features
            batch_data = list(self.data_buffer)
            X = self.prepare_batch_features(batch_data)
            
            # Make prediction
            prediction = self.model_predictor.model.predict(X.iloc[-1:])
            
            # Calculate confidence
            batch_predictions = self.model_predictor.model.predict(X)
            confidence = 1.0 - (np.std(batch_predictions) / abs(prediction[0]))
            
            # Get feature importance from the model's get_feature_importance method
            feature_importance = self.model_predictor.model.get_feature_importance()
            
            # Get top features
            top_features = dict(sorted(
                feature_importance.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:5])
            
            # Log prediction details
            logging.info(f"Prediction: {prediction[0]:.4f}")
            logging.info(f"Confidence: {confidence:.4f}")
            logging.info("Top features:")
            for feature, importance in top_features.items():
                logging.info(f"  {feature}: {importance:.4f}")
            
            return {
                'prediction': float(prediction[0]),
                'confidence': float(confidence),
                'is_confident': bool(confidence >= 0.8),
                'top_features': top_features
            }
            
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            raise

