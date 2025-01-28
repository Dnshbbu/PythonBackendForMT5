from model_implementations import ModelFactory, BaseTimeSeriesModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Optional
import sqlite3
import logging
import os
import json
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
import joblib

class TimeSeriesModelTrainer:
    def __init__(self, db_path: str, model_save_dir: str = 'models'):
        self.db_path = db_path
        self.model_save_dir = model_save_dir
        self.setup_logging()
        self.setup_directories()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def setup_directories(self):
        os.makedirs(self.model_save_dir, exist_ok=True)

    def load_data_from_db(self, table_name: str) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"""
                SELECT * FROM {table_name}
                ORDER BY Date, Time
            """
            
            df = pd.read_sql_query(query, conn, coerce_float=True)
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df = df.set_index('DateTime')
            
            return df
        finally:
            if conn:
                conn.close()

    def prepare_features_target(self, df: pd.DataFrame, 
                              target_col: str,
                              feature_cols: List[str],
                              prediction_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        # Create target variable with future values
        y = df[target_col].shift(-prediction_horizon)
        
        # Select and prepare features
        X = df[feature_cols].copy()
        X = X.fillna(0)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=feature_cols,
            index=X.index
        )
        
        # Remove rows where we don't have future data
        valid_indices = y.notna()
        X_scaled = X_scaled[valid_indices]
        y = y[valid_indices]
        
        return X_scaled, y

    def time_series_split(self, X: pd.DataFrame, y: pd.Series, 
                         n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return list(tscv.split(X))

    def train_and_save(self, 
                      table_name: str,
                      model_type: str,
                      target_col: str,
                      feature_cols: List[str],
                      prediction_horizon: int = 1,
                      model_params: Optional[Dict] = None) -> Tuple[str, Dict]:
        try:
            # Load and prepare data
            df = self.load_data_from_db(table_name)
            X, y = self.prepare_features_target(df, target_col, feature_cols, prediction_horizon)
            
            # Get model instance
            model = ModelFactory.get_model(model_type)
            model.feature_columns = feature_cols
            
            # Train model
            _, raw_metrics = model.train(X, y, **(model_params or {}))
            
            # Convert numpy types to Python native types in metrics
            metrics = {}
            for key, value in raw_metrics.items():
                if isinstance(value, dict):
                    # Handle nested dictionaries (like feature_importance)
                    metrics[key] = {k: float(v) if hasattr(v, 'dtype') else v 
                                  for k, v in value.items()}
                elif hasattr(value, 'dtype'):
                    # Convert numpy scalars to Python native types
                    metrics[key] = float(value)
                else:
                    metrics[key] = value
            
            # Save model and metadata
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = model.save(self.model_save_dir, timestamp)
            
            # Save scaler
            if self.scaler:
                scaler_path = os.path.join(
                    self.model_save_dir, 
                    f"{model_type}_{timestamp}_scaler.joblib"
                )
                joblib.dump(self.scaler, scaler_path)
            
            # Save feature configuration
            config = {
                'model_type': model_type,
                'features': feature_cols,
                'target': target_col,
                'prediction_horizon': prediction_horizon,
                'created_at': timestamp
            }
            
            config_path = os.path.join(
                self.model_save_dir, 
                f"{model_type}_{timestamp}_config.json"
            )
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

            # Save global feature configuration
            feature_config = {
                'features': feature_cols,
                'timestamp': datetime.now().isoformat(),
                'model_type': model_type,
                'latest_model': model_path
            }

            global_config_path = os.path.join(self.model_save_dir, 'feature_config.json')
            with open(global_config_path, 'w') as f:
                json.dump(feature_config, f, indent=4)
            
            return model_path, metrics
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise