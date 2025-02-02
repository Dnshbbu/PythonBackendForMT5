import os
import logging
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime
from model_predictor import ModelPredictor
import json
import sqlite3

class RealTimePricePredictor:
    """
    Real-time price prediction system with consistent feature usage.
    """
    
    def __init__(self, db_path: str, models_dir: str, training_manager=None, model_name: Optional[str] = None):
        """Initialize the predictor
        
        Args:
            db_path: Path to the SQLite database
            models_dir: Directory containing models
            training_manager: Optional training manager instance
            model_name: Optional specific model name to load
        """
        self.model_predictor = ModelPredictor(db_path, models_dir)
        self.models_dir = models_dir
        self.setup_logging()
        
        # Add data buffer for maintaining price history
        self.data_buffer = deque(maxlen=2)  # Keep current and previous data points
        
        self.training_manager = training_manager
        
        # Load specific model if provided, otherwise load latest
        if model_name:
            self.model_predictor.load_model_by_name(model_name)
            logging.info(f"Loaded specified model: {model_name}")
        else:
            self.model_predictor.load_latest_model()
            logging.info("Loaded latest model")
            
        # Get features from model repository
        self.selected_features = self.load_features_from_repository()
        
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

    def load_features_from_repository(self) -> List[str]:
        """Load features from model repository for current model"""
        try:
            if self.model_predictor.current_model_name:
                # Get features from repository for current model
                cursor = sqlite3.connect(self.model_predictor.db_path).cursor()
                cursor.execute("""
                    SELECT features 
                    FROM model_repository 
                    WHERE model_name = ?
                """, (self.model_predictor.current_model_name,))
                result = cursor.fetchone()
                
                if result and result[0]:
                    features = json.loads(result[0])
                    logging.info(f"Loaded {len(features)} features from repository for model {self.model_predictor.current_model_name}")
                    return features
            
            # Fallback to default features if repository lookup fails
            logging.warning("Using default feature list")
            return ['Price', 'Score', 'ExitScore']
            
        except Exception as e:
            logging.error(f"Error loading features from repository: {e}")
            return ['Price', 'Score', 'ExitScore']

    def load_model_by_name(self, model_name: str) -> None:
        """Load a specific model and its features
        
        Args:
            model_name: Name of the model to load
        """
        try:
            self.model_predictor.load_model_by_name(model_name)
            # Update features from repository for new model
            self.selected_features = self.load_features_from_repository()
            logging.info(f"Loaded model {model_name} with {len(self.selected_features)} features")
        except Exception as e:
            logging.error(f"Error loading model {model_name}: {e}")
            raise

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
            
            # Get feature importance based on model type
            try:
                # First try get_feature_importance() method (e.g., for XGBoost)
                feature_importance = self.model_predictor.model.get_feature_importance()
            except (AttributeError, Exception):
                try:
                    # Try feature_importances_ attribute (e.g., for RandomForest, DecisionTree)
                    importances = self.model_predictor.model.feature_importances_
                    feature_importance = dict(zip(self.selected_features, importances))
                except (AttributeError, Exception) as e:
                    logging.warning(f"Could not get feature importance: {e}")
                    # Fallback to equal importance if no method available
                    feature_importance = {feature: 1.0/len(self.selected_features) 
                                       for feature in self.selected_features}
            
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
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_type': type(self.model_predictor.model).__name__
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            raise

    def add_data_point(self, data_point: Dict) -> Optional[Dict]:
        """Process single data point and return prediction for next price"""
        try:
            # # Check and reload model if needed
            # if self.training_manager is not None:
            #     try:
            #         training_status = self.training_manager.get_latest_training_status()
            #         if training_status.get('status') == 'completed':
            #             latest_model = training_status.get('model_path')
            #             if latest_model and latest_model != self.model_predictor.current_model_name:
            #                 logging.info(f"New model available, reloading: {latest_model}")
            #                 self.model_predictor.load_model_by_name(latest_model)
            #                 # Update features for new model
            #                 self.selected_features = self.load_features_from_repository()
            #     except Exception as e:
            #         logging.warning(f"Error checking training status: {e}")

            # Rest of the method remains the same
            processed_data = self.process_raw_data(data_point)
            current_price = float(data_point.get('Price', 0))
            
            self.data_buffer.append({
                'processed_data': processed_data,
                'price': current_price,
                'timestamp': data_point.get('Time')
            })
            
            if len(self.data_buffer) >= 1:
                features = self.prepare_features(processed_data)
                prediction_result = self.make_prediction(features)
                
                if len(self.data_buffer) == 2:
                    previous_price = self.data_buffer[0]['price']
                    price_change = current_price - previous_price
                    prediction_result.update({
                        'current_price': current_price,
                        'previous_price': previous_price,
                        'price_change': price_change,
                    })
                
                return prediction_result
                
        except Exception as e:
            logging.error(f"Error in add_data_point: {e}")
            raise

