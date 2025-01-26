"""
real_time_ml.py - Real-time ML training and prediction system
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import queue
import threading
import json
from typing import Dict, List, Optional
import logging
from xgboost import XGBRegressor
import joblib
from sklearn.preprocessing import RobustScaler



class RealTimeMLProcessor:
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.data_buffer = queue.Queue(maxsize=buffer_size)
        self.processed_data = pd.DataFrame()
        self.model = None
        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        self.lock = threading.Lock()
        self.setup_logging()
        
        # Create necessary directories
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.predictions_dir = os.path.join(self.base_dir, 'predictions')
        os.makedirs(self.predictions_dir, exist_ok=True)
        
        # Initialize model
        self._initialize_model()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('real_time_ml.log'),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_model(self):
        """Initialize or load existing model"""
        self.model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Try to load existing model
        model_path = os.path.join(self.base_dir, 'models', 'latest_real_time_model.joblib')
        if os.path.exists(model_path):
            try:
                saved_state = joblib.load(model_path)
                self.model = saved_state['model']
                self.feature_scaler = saved_state['feature_scaler']
                self.target_scaler = saved_state['target_scaler']
                logging.info("Loaded existing model successfully")
            except Exception as e:
                logging.error(f"Error loading existing model: {e}")
    
    def process_new_data(self, data: Dict) -> None:
        """Process incoming data from ZMQ server"""
        try:
            # Data is already a dictionary, no need to parse
            if not data:
                logging.warning("Received empty data")
                return

            # Add to buffer
            if self.data_buffer.full():
                self.data_buffer.get()  # Remove oldest item if buffer is full
            self.data_buffer.put(data)
            
            # Update processed data
            with self.lock:
                new_row_df = pd.DataFrame([data])
                self.processed_data = pd.concat([self.processed_data, new_row_df], ignore_index=True)
                
                # Keep only last buffer_size rows
                if len(self.processed_data) > self.buffer_size:
                    self.processed_data = self.processed_data.iloc[-self.buffer_size:]
            
            # Make prediction for new data
            prediction = self.predict_single(data)
            self._save_prediction(data, prediction)
            
            # Retrain model if enough new data
            if len(self.processed_data) >= 100:  # Minimum rows for training
                self.retrain_model()
        
        except Exception as e:
            logging.error(f"Error processing new data: {str(e)}")
            raise
    
    def _prepare_training_data(self, data: pd.DataFrame) -> tuple:
        """Prepare data for training"""
        try:
            # Extract features from factors and scores
            feature_columns = []
            
            # Entry Score features
            entry_score_cols = [col for col in data.columns if col.startswith('EntryScore_')]
            feature_columns.extend(entry_score_cols)
            
            # Factor features
            factor_cols = [col for col in data.columns if col.startswith('Factors_')]
            feature_columns.extend(factor_cols)
            
            # Ensure we have the minimum required features
            if not feature_columns:
                raise ValueError("No valid feature columns found in data")
            
            # Select features and target
            X = data[feature_columns]
            y = data['Price']  # Using Price as target
            
            # Log the feature columns being used
            logging.info(f"Using features: {feature_columns}")
            
            return X, y
            
        except Exception as e:
            logging.error(f"Error preparing training data: {e}")
            raise
    
    def predict_single(self, data: Dict) -> float:
        """Make prediction for single data point"""
        try:
            if self.model is None:
                logging.warning("Model not initialized yet")
                return None
            
            # Convert data to DataFrame
            df = pd.DataFrame([data])
            
            # Prepare features
            X, _ = self._prepare_training_data(df)
            
            if X.empty:
                logging.warning("No valid features found for prediction")
                return None
            
            # Scale features
            X_scaled = self.feature_scaler.transform(X)
            
            # Make prediction
            pred_scaled = self.model.predict(X_scaled)
            prediction = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
            
            return prediction
        
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            return None
    
    def _save_prediction(self, data: Dict, prediction: float) -> None:
        """Save prediction to CSV file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            predictions_file = os.path.join(self.predictions_dir, f'predictions_{datetime.now().strftime("%Y%m%d")}.csv')
            
            # Prepare row data
            row_data = {
                'Timestamp': timestamp,
                'Actual_Price': data.get('Price', ''),
                'Predicted_Price': prediction,
                **data  # Include all original data
            }
            
            # Save to CSV
            pd.DataFrame([row_data]).to_csv(
                predictions_file, 
                mode='a', 
                header=not os.path.exists(predictions_file), 
                index=False
            )
            
        except Exception as e:
            logging.error(f"Error saving prediction: {e}")
            raise
    
    def retrain_model(self) -> None:
        """Retrain model with current data"""
        try:
            with self.lock:
                if len(self.processed_data) < 100:
                    return
                
                # Prepare features and target
                X, y = self._prepare_training_data(self.processed_data)
                
                if X.empty or y.empty:
                    logging.warning("No valid data for training")
                    return
                
                # Scale data
                X_scaled = self.feature_scaler.fit_transform(X)
                y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1))
                
                # Train model
                self.model.fit(X_scaled, y_scaled.ravel())
                
                # Save model
                self._save_model()
                
                logging.info("Model retrained successfully")
        
        except Exception as e:
            logging.error(f"Error retraining model: {e}")
            raise


    def _save_model(self) -> None:
        """Save current model state"""
        try:
            model_dir = os.path.join(self.base_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)
            
            save_path = os.path.join(model_dir, 'latest_real_time_model.joblib')
            
            model_state = {
                'model': self.model,
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler,
                'timestamp': datetime.now().isoformat()
            }
            
            joblib.dump(model_state, save_path)
            logging.info(f"Model saved successfully to {save_path}")
            
        except Exception as e:
            logging.error(f"Error saving model: {e}")
    
    def get_latest_predictions(self, n: int = 100) -> pd.DataFrame:
        """Get latest n predictions for display"""
        try:
            latest_file = max(
                [f for f in os.listdir(self.predictions_dir) if f.startswith('predictions_')],
                key=lambda x: os.path.getctime(os.path.join(self.predictions_dir, x))
            )
            
            predictions_df = pd.read_csv(os.path.join(self.predictions_dir, latest_file))
            return predictions_df.tail(n)
            
        except Exception as e:
            logging.error(f"Error getting latest predictions: {e}")
            return pd.DataFrame()
        


