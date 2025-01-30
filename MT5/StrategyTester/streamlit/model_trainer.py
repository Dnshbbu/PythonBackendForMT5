import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Optional, Any
import sqlite3
import logging
import joblib
import os
import json
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit

class TimeSeriesModelTrainer:
    def __init__(self, db_path: str, models_dir: str):
        """
        Initialize the trainer with database and model paths
        
        Args:
            db_path: Path to SQLite database
            models_dir: Directory to save trained models
        """
        self.db_path = db_path
        self.models_dir = models_dir
        self.setup_logging()
        self.setup_directories()

    def setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.models_dir, exist_ok=True)

    def validate_temporal_order(self, df: pd.DataFrame) -> bool:
        """
        Validate that the DataFrame is properly ordered in time
        
        Args:
            df: DataFrame with DateTime index
            
        Returns:
            bool: True if ordering is valid
            
        Raises:
            ValueError: If temporal ordering is invalid
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
            
        if not df.index.is_monotonic_increasing:
            violations = df.index[:-1][df.index[1:] <= df.index[:-1]]
            if len(violations) > 0:
                raise ValueError(
                    f"Data not in temporal order. Violations found at: {violations}"
                )
        return True

    def load_data_from_db(self, table_name: str) -> pd.DataFrame:
        """Load data from SQLite database with strict temporal ordering"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"""
                SELECT * FROM {table_name}
                ORDER BY Date, Time
            """
            
            df = pd.read_sql_query(query, conn, coerce_float=True)
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df = df.set_index('DateTime')
            
            self.validate_temporal_order(df)
            logging.info(f"Loaded {len(df)} rows from {table_name}")
            
            return df
        finally:
            if conn:
                conn.close()

    def load_data_from_multiple_tables(self, table_names: List[str]) -> pd.DataFrame:
        """
        Load and combine data from multiple tables with proper ordering
        
        Args:
            table_names: List of table names to load data from
            
        Returns:
            Combined DataFrame with data from all tables
        """
        try:
            all_data = []
            
            for table_name in table_names:
                conn = sqlite3.connect(self.db_path)
                query = f"""
                    SELECT * FROM {table_name}
                    ORDER BY Date, Time
                """
                
                df = pd.read_sql_query(query, conn, coerce_float=True)
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                df['source_table'] = table_name  # Track source table
                df = df.set_index('DateTime')
                all_data.append(df)
                
                conn.close()
            
            # Combine all data and sort by datetime
            if not all_data:
                raise ValueError("No data loaded from any table")
                
            combined_df = pd.concat(all_data, axis=0)
            combined_df = combined_df.sort_index()
            
            # Validate temporal ordering
            self.validate_temporal_order(combined_df)
            
            logging.info(f"Loaded data from {len(table_names)} tables")
            logging.info(f"Total rows: {len(combined_df)}")
            logging.info(f"Time range: {combined_df.index.min()} to {combined_df.index.max()}")
            
            return combined_df
            
        except Exception as e:
            logging.error(f"Error loading data from multiple tables: {e}")
            raise

    def prepare_features_target(self, df: pd.DataFrame, 
                              target_col: str,
                              feature_cols: List[str],
                              prediction_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target with strict temporal considerations"""
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
        """Create time series cross-validation splits"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return list(tscv.split(X))


    def train_model(self, X: pd.DataFrame, y: pd.Series, 
               model_params: Optional[Dict] = None,
               existing_model: Optional[Any] = None) -> Tuple[Any, Dict]:
        """
        Train model with support for incremental learning
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_params: Model parameters dictionary
            existing_model: Optional existing model for incremental training
            
        Returns:
            Tuple of (model, metrics)
        """
        try:
            from xgboost import XGBRegressor, DMatrix
            
            if model_params is None:
                model_params = {
                    'objective': 'reg:squarederror',
                    'n_estimators': 100,  # Base number of trees
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'early_stopping_rounds': 10
                }
            
            if existing_model is not None and isinstance(existing_model, XGBRegressor):
                logging.info("Performing incremental training with existing model")
                
                # Get current number of trees
                current_n_trees = existing_model.n_estimators
                
                # Set number of new trees to add
                n_additional_trees = 50  # You can adjust this number
                model_params['n_estimators'] = n_additional_trees
                
                # Create DMatrix for training
                dtrain = DMatrix(X, label=y)
                
                # Get existing model's booster
                existing_booster = existing_model.get_booster()
                
                # Continue training from the existing model
                model = XGBRegressor(**model_params)
                model.fit(X, y,
                        xgb_model=existing_booster,  # Use existing model as starting point
                        verbose=False)
                
                training_type = 'incremental'
                total_trees = current_n_trees + n_additional_trees
                
            else:
                logging.info("Performing full training (no valid existing model)")
                model = XGBRegressor(**model_params)
                model.fit(X, y, verbose=False)
                training_type = 'full'
                total_trees = model_params['n_estimators']
            
            # Calculate metrics
            predictions = model.predict(X)
            mse = np.mean((y - predictions) ** 2)
            r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
            
            metrics = {
                'rmse': float(np.sqrt(mse)),
                'r2': float(r2),
                'training_samples': len(X),
                'training_type': training_type,
                'n_trees': total_trees,
                'training_period': {
                    'start': X.index.min().isoformat(),
                    'end': X.index.max().isoformat()
                }
            }
            
            return model, metrics
        
        except Exception as e:
            logging.error(f"Error in train_model: {str(e)}")
            logging.exception("Detailed traceback:")
            raise


    def save_model_and_metadata(self, model: Any, 
                              feature_cols: List[str],
                              metrics: Dict,
                              model_name: Optional[str] = None) -> str:
        """Save model and associated metadata"""
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"model_{timestamp}"
            
        # Save model
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        
        # Save scaler if available
        if hasattr(self, 'scaler'):
            scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
        
        # Save feature configuration
        feature_config = {
            'features': feature_cols,
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path
        }
        
        config_path = os.path.join(self.models_dir, f"{model_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(feature_config, f, indent=4)
            
        # Save metrics
        metrics_path = os.path.join(self.models_dir, f"{model_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        return model_path



    def train_and_save_multi_table(self, 
                                table_names: List[str],
                                target_col: str,
                                prediction_horizon: int = 1,
                                feature_cols: Optional[List[str]] = None,
                                model_params: Optional[Dict] = None,
                                model_name: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Train model using data from multiple tables with support for incremental learning
        """
        try:
            # Load and combine data from all tables
            df = self.load_data_from_multiple_tables(table_names)
            logging.info(f"Combined data shape: {df.shape}")
            
            # Prepare features and target
            X, y = self.prepare_features_target(
                df, 
                target_col,
                feature_cols=feature_cols,
                prediction_horizon=prediction_horizon
            )
            logging.info(f"Prepared features shape: {X.shape}, target shape: {y.shape}")
            
            # Try to load existing model
            existing_model = None
            if model_name:
                model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
                if os.path.exists(model_path):
                    try:
                        existing_model = joblib.load(model_path)
                        logging.info(f"Successfully loaded existing model: {model_path}")
                    except Exception as e:
                        logging.warning(f"Could not load existing model: {e}")
                        existing_model = None

            # Train or update model
            model, metrics = self.train_model(
                X=X,
                y=y,
                model_params=model_params,
                existing_model=existing_model
            )
            
            # Add additional metrics
            metrics.update({
                'training_tables': table_names,
                'training_time': datetime.now().isoformat(),
                'data_points': len(X),
                'features_used': list(X.columns),
                'model_name': model_name,
                'training_period': {
                    'start': str(X.index.min()),
                    'end': str(X.index.max())
                }
            })
            
            # Save model with same name if updating, or generate new name if new model
            if not model_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"model_{timestamp}"
                
            model_path = self.save_model_and_metadata(
                model=model,
                feature_cols=X.columns.tolist(),
                metrics=metrics,
                model_name=model_name
            )
            logging.info(f"Model saved to: {model_path}")
            
            # Save training history
            try:
                self._update_training_history(model_path, table_names, metrics)
            except Exception as history_error:
                logging.warning(f"Error updating training history: {history_error}")
            
            return model_path, metrics
            
        except Exception as e:
            logging.error(f"Error in multi-table training pipeline: {str(e)}")
            logging.exception("Detailed traceback:")
            raise

    def _update_training_history(self, model_path: str, table_names: List[str], metrics: Dict):
        """Helper method to update training history"""
        history_path = os.path.join(self.models_dir, 'training_history.json')
        
        # Prepare new history entry
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'tables_used': table_names,
            'model_path': model_path,
            'metrics': {
                k: str(v) if isinstance(v, (np.floating, np.integer)) else v 
                for k, v in metrics.items()
            }
        }
        
        # Load existing history or create new
        try:
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    try:
                        history = json.load(f)
                    except json.JSONDecodeError:
                        history = {'training_events': []}
            else:
                history = {'training_events': []}
            
            # Add new entry
            history['training_events'].append(history_entry)
            
            # Save updated history
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)
            logging.info("Training history updated successfully")
            
        except Exception as e:
            logging.warning(f"Failed to update training history: {e}")
            logging.exception("History update error details:")

        