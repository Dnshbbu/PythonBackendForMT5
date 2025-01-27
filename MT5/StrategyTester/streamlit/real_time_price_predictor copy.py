import os
import logging
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from collections import deque
from model_predictor import ModelPredictor

class RealTimePricePredictor:
    def __init__(self, db_path: str, models_dir: str, batch_size: int = 10):
        self.model_predictor = ModelPredictor(db_path, models_dir)
        self.batch_size = batch_size
        self.data_buffer = deque(maxlen=batch_size)
        self.last_prediction = None
        self.setup_logging()
        
        # Define columns that need splitting
        self.split_columns = ['Factors', 'ExitFactors', 'EntryScore', 'ExitScoreDetails', 'Pullback']
        
        logging.info(f"Initialized RealTimePricePredictor with batch size: {batch_size}")
        if self.model_predictor.model:
            logging.info("Model loaded successfully")
            logging.info(f"Number of features: {len(self.model_predictor.feature_columns)}")
            logging.info(f"Features: {self.model_predictor.feature_columns}")

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

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
                
                # Create column name like 'Factors_maScore'
                new_key = f"{column_name}_{key}"
                
                # Try to convert to float if possible
                try:
                    result[new_key] = float(value)
                except ValueError:
                    # Handle boolean values
                    if value.lower() == 'true':
                        result[new_key] = True
                    elif value.lower() == 'false':
                        result[new_key] = False
                    else:
                        result[new_key] = value
                        
        return result

    def process_raw_data(self, data_point: Dict) -> Dict:
        """Process raw data point into proper format for model"""
        try:
            processed_data = {}
            
            # Process basic numeric fields
            numeric_fields = ['Price', 'Equity', 'Balance', 'Profit', 'Score', 'ExitScore', 'Positions']
            for field in numeric_fields:
                try:
                    processed_data[field] = float(data_point.get(field, 0))
                except (ValueError, TypeError):
                    processed_data[field] = 0.0
            
            # Process string fields that need splitting
            for column in self.split_columns:
                if column in data_point:
                    split_data = self.split_factor_string(column, data_point[column])
                    processed_data.update(split_data)
            
            # Add Date and Time if present
            if 'Date' in data_point:
                processed_data['Date'] = data_point['Date']
            if 'Time' in data_point:
                processed_data['Time'] = data_point['Time']
            
            # Log the number of features processed
            logging.info(f"Processed data point with {len(processed_data)} features")
            logging.debug(f"Features: {list(processed_data.keys())}")
            
            return processed_data
            
        except Exception as e:
            logging.error(f"Error processing raw data: {e}")
            raise


    # def prepare_batch_features(self, batch_data: List[Dict]) -> pd.DataFrame:
    #     """Prepare features from batch data"""
    #     try:
    #         # Convert batch to DataFrame
    #         df = pd.DataFrame(batch_data)
            
    #         # Log initial data shape
    #         logging.info(f"Initial batch data shape: {df.shape}")
            
    #         # Get required features from model
    #         required_features = self.model_predictor.feature_columns
            
    #         # Check for missing features
    #         missing_features = [f for f in required_features if f not in df.columns]
    #         if missing_features:
    #             logging.warning(f"Missing features: {missing_features}")
    #             for feature in missing_features:
    #                 df[feature] = 0.0
            
    #         # Select only required features
    #         X = df[required_features].copy()
            
    #         # Convert all features to float where possible
    #         for col in X.columns:
    #             try:
    #                 X[col] = X[col].astype(float)
    #             except Exception as e:
    #                 logging.warning(f"Could not convert {col} to float: {e}")
            
    #         # Apply scaling if available
    #         if self.model_predictor.scaler:
    #             X = pd.DataFrame(
    #                 self.model_predictor.scaler.transform(X),
    #                 columns=required_features,
    #                 index=X.index
    #             )
    #             logging.debug("Applied feature scaling")
            
    #         # Log feature statistics
    #         logging.info(f"Final feature matrix shape: {X.shape}")
    #         return X
            
    #     except Exception as e:
    #         logging.error(f"Error preparing batch features: {e}")
    #         raise


    def prepare_batch_features(self, batch_data: List[Dict]) -> pd.DataFrame:
        """Prepare features from batch data more efficiently"""
        try:
            # Convert batch to DataFrame
            df = pd.DataFrame(batch_data)
            
            # Log initial data shape
            logging.info(f"Initial batch data shape: {df.shape}")
            
            # Get required features from model
            required_features = self.model_predictor.feature_columns
            
            # Create DataFrame for missing features
            missing_features = [f for f in required_features if f not in df.columns]
            if missing_features:
                logging.warning(f"Missing features: {missing_features}")
                # Create a DataFrame with zeros for missing features
                missing_df = pd.DataFrame(
                    0.0, 
                    index=df.index, 
                    columns=missing_features
                )
                # Concatenate with original DataFrame
                df = pd.concat([df, missing_df], axis=1)
            
            # Select only required features
            X = df[required_features].copy()
            
            # Convert all features to float where possible
            for col in X.columns:
                try:
                    X[col] = X[col].astype(float)
                except Exception as e:
                    logging.warning(f"Could not convert {col} to float: {e}")
            
            # Apply scaling if available
            if self.model_predictor.scaler:
                X = pd.DataFrame(
                    self.model_predictor.scaler.transform(X),
                    columns=required_features,
                    index=X.index
                )
                logging.debug("Applied feature scaling")
            
            # Log feature statistics
            logging.info(f"Final feature matrix shape: {X.shape}")
            return X
            
        except Exception as e:
            logging.error(f"Error preparing batch features: {e}")
            raise

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
            logging.info(f"Current price: {current_price}")
            
            # Make prediction if buffer is full
            if len(self.data_buffer) == self.batch_size:
                prediction_result = self.make_prediction()
                
                # Log prediction change
                if self.last_prediction is not None:
                    price_change = current_price - float(data_point.get('Price', 0))
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
            
            # Get feature importance
            feature_importance = {
                str(col): float(imp) 
                for col, imp in zip(X.columns, self.model_predictor.model.feature_importances_)
            }
            
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