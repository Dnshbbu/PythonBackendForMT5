import logging
import sqlite3
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import os

class MetaModelTrainer:
    def __init__(self, db_path: str, models_dir: str):
        """
        Initialize the meta-model trainer
        
        Args:
            db_path: Path to the SQLite database
            models_dir: Directory to store trained models
        """
        self.db_path = db_path
        self.models_dir = models_dir
        self.meta_model = None
        self.feature_scaler = None
        self.setup_logging()
        self.setup_mlflow()
        
    def setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def setup_mlflow(self):
        """Configure MLflow settings"""
        # Set MLflow tracking URI to SQLite database
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Create or get experiment
        experiment_name = "meta_model_experiment"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            mlflow.create_experiment(experiment_name)
            
        mlflow.set_experiment(experiment_name)
        logging.info(f"Using MLflow experiment: {experiment_name}")
        
    def load_historical_predictions(self, start_date: datetime, end_date: datetime, 
                                 run_ids: List[str]) -> pd.DataFrame:
        """
        Load historical predictions from specified run_ids
        
        Args:
            start_date: Start date for loading predictions
            end_date: End date for loading predictions
            run_ids: List of run_ids to load predictions from
            
        Returns:
            DataFrame containing predictions from all models
        """
        try:
            query = """
                SELECT 
                    datetime,
                    actual_price,
                    predicted_price,
                    error,
                    price_change,
                    predicted_change,
                    price_volatility,
                    model_name,
                    run_id
                FROM historical_predictions
                WHERE datetime BETWEEN ? AND ?
                AND run_id IN ({})
                ORDER BY datetime
            """.format(','.join('?' * len(run_ids)))
            
            params = [start_date.isoformat(), end_date.isoformat()] + run_ids
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params, parse_dates=['datetime'])
            
            if df.empty:
                raise ValueError(f"No predictions found for the specified run_ids and date range")
            
            # Create features for each model
            pivot_df = pd.pivot_table(
                df,
                index='datetime',
                columns='run_id',
                values=['predicted_price', 'error', 'predicted_change'],
                aggfunc='first'
            ).reset_index()
            
            # Flatten column names
            pivot_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                              for col in pivot_df.columns]
            
            # Add the actual values (same for all run_ids)
            base_features = df.groupby('datetime').agg({
                'actual_price': 'first',
                'price_change': 'first',
                'price_volatility': 'first'
            }).reset_index()
            
            # Merge all features
            final_df = pd.merge(base_features, pivot_df, on='datetime')
            
            logging.info(f"Loaded {len(final_df)} historical predictions")
            return final_df
            
        except Exception as e:
            logging.error(f"Error loading historical predictions: {e}")
            raise
            
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for training the meta-model
        
        Args:
            data: DataFrame containing predictions and actual values
            
        Returns:
            Tuple of (X, y) where X is the feature matrix and y is the target vector
        """
        # Select feature columns (all except datetime and actual_price)
        feature_cols = [col for col in data.columns 
                       if col not in ['datetime', 'actual_price']]
        
        X = data[feature_cols].values
        y = data['actual_price'].values
        
        # Scale features
        self.feature_scaler = StandardScaler()
        X = self.feature_scaler.fit_transform(X)
        
        return X, y
        
    def train_meta_model(self, start_date: datetime, end_date: datetime, 
                        run_ids: List[str], test_size: float = 0.2) -> Dict[str, float]:
        """
        Train a meta-model using predictions from multiple models
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            run_ids: List of run_ids to use for training
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Load and prepare data
            data = self.load_historical_predictions(start_date, end_date, run_ids)
            X, y = self.prepare_features(data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )
            
            # Initialize and train meta-model (XGBoost)
            self.meta_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            
            # Start MLflow run
            with mlflow.start_run(run_name='meta_model'):
                # Log parameters
                mlflow.log_params({
                    'test_size': test_size,
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3
                })
                
                # Train model
                self.meta_model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    verbose=False
                )
                
                # Make predictions
                train_pred = self.meta_model.predict(X_train)
                test_pred = self.meta_model.predict(X_test)
                
                # Calculate metrics
                metrics = {
                    'train_rmse': np.sqrt(np.mean((y_train - train_pred) ** 2)),
                    'test_rmse': np.sqrt(np.mean((y_test - test_pred) ** 2)),
                    'train_mae': np.mean(np.abs(y_train - train_pred)),
                    'test_mae': np.mean(np.abs(y_test - test_pred))
                }
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Save model
                model_path = os.path.join(self.models_dir, 'meta_model.json')
                self.meta_model.save_model(model_path)
                mlflow.log_artifact(model_path)
                
                logging.info("Meta-model training completed successfully")
                return metrics
                
        except Exception as e:
            logging.error(f"Error training meta-model: {e}")
            raise
            
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained meta-model
        
        Args:
            features: DataFrame containing features for prediction
            
        Returns:
            Array of predictions
        """
        if self.meta_model is None:
            raise ValueError("Meta-model has not been trained yet")
            
        # Prepare features
        X = self.feature_scaler.transform(features)
        
        # Make predictions
        predictions = self.meta_model.predict(X)
        return predictions 