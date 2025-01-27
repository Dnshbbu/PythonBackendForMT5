import os
import logging
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime
from model_predictor import ModelPredictor

class RealTimePricePredictor:
    """
    Real-time price prediction system with efficient feature engineering and batch processing.
    
    This class handles:
    - Real-time data processing
    - Feature engineering (time-based, lagged, and rolling features)
    - Batch prediction management
    - Efficient DataFrame operations
    """
    
    def __init__(self, db_path: str, models_dir: str, batch_size: int = 10):
        """
        Initialize the predictor with model and data management settings.
        
        Args:
            db_path: Path to the database
            models_dir: Directory containing trained models
            batch_size: Number of data points to collect before making predictions
        """
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
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def split_factor_string(self, column_name: str, factor_string: str) -> Dict[str, any]:
        """
        Split a factor string into individual components
        
        Args:
            column_name: Name of the column containing factors
            factor_string: String containing factor data
            
        Returns:
            Dictionary of parsed factors
        """
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

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from Date and Time columns"""
        if 'Date' in df.columns and 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df['Hour'] = df['DateTime'].dt.hour
            df['DayOfWeek'] = df['DateTime'].dt.dayofweek
            df['DayOfMonth'] = df['DateTime'].dt.day
            df['WeekOfYear'] = df['DateTime'].dt.isocalendar().week
            df['Month'] = df['DateTime'].dt.month
            df['Quarter'] = df['DateTime'].dt.quarter
            
            # Market sessions
            df['IsAsiaSession'] = ((df['Hour'] >= 0) & (df['Hour'] < 8)).astype(int)
            df['IsEuropeSession'] = ((df['Hour'] >= 8) & (df['Hour'] < 16)).astype(int)
            df['IsUSSession'] = ((df['Hour'] >= 13) & (df['Hour'] < 21)).astype(int)
        return df

    def create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        lag_periods = [1, 5, 10, 20, 50]
        
        lagged_features = {}
        for col in numeric_cols:
            if col not in ['Hour', 'DayOfWeek', 'Month', 'Quarter']:  # Exclude time features
                for lag in lag_periods:
                    lagged_features[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Concatenate all lagged features at once
        if lagged_features:
            lagged_df = pd.DataFrame(lagged_features, index=df.index)
            df = pd.concat([df, lagged_df], axis=1)
        
        return df




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
    #                 price_change = current_price - float(data_point.get('Price', 0))
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
                prediction_result = self.make_prediction()
                
                # Log prediction change
                if self.last_prediction is not None:
                    # Get previous price from the second-to-last item in buffer
                    previous_data = list(self.data_buffer)[-2]  # Get second to last item
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



    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features for specified columns"""
        try:
            feature_cols = ['Price', 'Score', 'ExitScore']
            factor_cols = [col for col in df.columns if 'Factors_' in col]
            feature_cols.extend(factor_cols)
            
            windows = [5, 10, 20, 50, 100]
            rolling_features = {}
            
            for col in feature_cols:
                if col in df.columns:
                    # First try to convert column to numeric
                    try:
                        series = pd.to_numeric(df[col], errors='coerce')
                        if not series.isna().all():  # Only process if we have some valid numbers
                            for window in windows:
                                rolling = series.rolling(window=window, min_periods=1)
                                rolling_features[f'{col}_rolling_mean_{window}'] = rolling.mean()
                                rolling_features[f'{col}_rolling_std_{window}'] = rolling.std()
                                rolling_features[f'{col}_rolling_min_{window}'] = rolling.min()
                                rolling_features[f'{col}_rolling_max_{window}'] = rolling.max()
                                rolling_features[f'{col}_momentum_{window}'] = series.diff(window)
                    except Exception as e:
                        logging.warning(f"Could not create rolling features for column {col}: {str(e)}")
                        continue
            
            # Concatenate all rolling features at once
            if rolling_features:
                rolling_df = pd.DataFrame(rolling_features, index=df.index)
                df = pd.concat([df, rolling_df], axis=1)
            
            return df
            
        except Exception as e:
            logging.error(f"Error in create_rolling_features: {str(e)}")
            return df  # Return original DataFrame if there's an error

    def process_raw_data(self, data_point: Dict) -> Dict:
        """Process raw data point into proper format for model"""
        try:
            processed_data = {}
            
            # Process basic numeric fields
            numeric_fields = ['Price', 'Equity', 'Balance', 'Profit', 'Score', 'ExitScore', 'Positions']
            for field in numeric_fields:
                try:
                    value = data_point.get(field, '0')
                    processed_data[field] = float(value) if value != '' else 0.0
                except (ValueError, TypeError):
                    processed_data[field] = 0.0
            
            # Process string fields that need splitting
            for column in self.split_columns:
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
            
            # Add Date and Time if present
            if 'Date' in data_point:
                processed_data['Date'] = data_point['Date']
            if 'Time' in data_point:
                processed_data['Time'] = data_point['Time']
            
            return processed_data
            
        except Exception as e:
            logging.error(f"Error processing raw data: {e}")
            raise

    def prepare_batch_features(self, batch_data: List[Dict]) -> pd.DataFrame:
        """Prepare features from batch data with proper feature engineering"""
        try:
            # Convert batch to DataFrame
            df = pd.DataFrame(batch_data)
            
            # Log initial data shape
            logging.info(f"Initial batch data shape: {df.shape}")
            
            # Create all required features
            df = self.create_time_features(df)
            df = self.create_lagged_features(df)
            df = self.create_rolling_features(df)
            
            # Get required features from model
            required_features = self.model_predictor.feature_columns
            
            # Handle missing features efficiently
            missing_features = [f for f in required_features if f not in df.columns]
            if missing_features:
                logging.warning(f"Missing features after generation: {missing_features}")
                missing_df = pd.DataFrame(0.0, index=df.index, columns=missing_features)
                df = pd.concat([df, missing_df], axis=1)
            
            # Select only required features and handle data types
            X = df[required_features].copy()
            
            # Convert all columns to numeric, replacing errors with 0
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
            # Apply scaling if available
            if self.model_predictor.scaler:
                X = pd.DataFrame(
                    self.model_predictor.scaler.transform(X),
                    columns=required_features,
                    index=X.index
                )
            
            logging.info(f"Final feature matrix shape: {X.shape}")
            return X
            
        except Exception as e:
            logging.error(f"Error preparing batch features: {e}")
            raise









