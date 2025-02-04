import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from typing import List, Tuple, Dict, Optional
import sqlite3
import logging
import joblib
import os
import json
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit

class TimeSeriesXGBoostTrainer:
    def __init__(self, db_path: str, model_save_dir: str = 'models'):
        """
        Initialize the trainer with database and model paths
        
        Args:
            db_path: Path to SQLite database
            model_save_dir: Directory to save trained models
        """
        self.db_path = db_path
        self.model_save_dir = model_save_dir
        self.setup_logging()
        self.setup_directories()
        
        # Default model parameters
        self.model_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1
        }
        
    def setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.model_save_dir, exist_ok=True)

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
            # Find where ordering is violated
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
            # Order by date and time in the SQL query itself
            query = f"""
                SELECT * FROM {table_name}
                ORDER BY Date, Time
            """
            
            df = pd.read_sql_query(query, conn, coerce_float=True)
            
            # Convert to datetime and set as index
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df = df.set_index('DateTime')
            
            # Validate temporal ordering
            self.validate_temporal_order(df)
            
            # Convert numeric columns
            numeric_columns = [
                'Price', 'Equity', 'Balance', 'Profit', 'Positions', 
                'Score', 'ExitScore'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logging.info(f"Loaded {len(df)} rows from {table_name}")
            logging.info(f"Time range: {df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading data from database: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def prepare_features_target(self, df: pd.DataFrame, 
                              target_col: str,
                              feature_cols: Optional[List[str]] = None,
                              prediction_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target with strict temporal considerations"""
        try:
            # Validate temporal ordering
            self.validate_temporal_order(df)
            
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in DataFrame")
            
            # Get feature columns if not provided
            if feature_cols is None:
                raise ValueError("Feature columns must be specified")
            
            # Create target variable with future values
            y = df[target_col].shift(-prediction_horizon)
            
            # Handle missing features
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                logging.warning(f"Missing columns: {missing_cols}")
                for col in missing_cols:
                    df[col] = 0
            
            # Select features
            X = df[feature_cols].copy()
            
            # Fill any NaN values in features
            X = X.fillna(0)
            
            # Create and save scaler
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
            
            # Additional temporal validation
            if len(X_scaled) == 0:
                raise ValueError("No valid data points after temporal alignment")
            
            logging.info(f"Training data time range: {X_scaled.index.min()} to {X_scaled.index.max()}")
            logging.info(f"Feature shape: {X_scaled.shape}, Target shape: {y.shape}")
            
            return X_scaled, y
            
        except Exception as e:
            logging.error(f"Error in prepare_features_target: {e}")
            raise

    def time_series_split(self, X: pd.DataFrame, y: pd.Series, 
                         n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create time series cross-validation splits with validation"""
        try:
            # Validate temporal ordering again
            self.validate_temporal_order(X)
            
            tscv = TimeSeriesSplit(n_splits=n_splits)
            splits = []
            
            for train_idx, val_idx in tscv.split(X):
                # Validate that validation indices come after training indices
                train_max_time = X.index[train_idx].max()
                val_min_time = X.index[val_idx].min()
                
                if val_min_time <= train_max_time:
                    raise ValueError(
                        f"Temporal leakage detected in fold: "
                        f"val_min_time ({val_min_time}) <= train_max_time ({train_max_time})"
                    )
                
                splits.append((train_idx, val_idx))
                
                logging.info(
                    f"Split - Train: {X.index[train_idx].min()} to {X.index[train_idx].max()}, "
                    f"Val: {X.index[val_idx].min()} to {X.index[val_idx].max()}"
                )
            
            return splits
            
        except Exception as e:
            logging.error(f"Error in time_series_split: {e}")
            raise

    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   model_params: Optional[Dict] = None,
                   n_splits: int = 5) -> Tuple[xgb.XGBRegressor, Dict]:
        """Train model with strict temporal validation"""
        try:
            if model_params is None:
                model_params = self.model_params.copy()
            
            # Get time series splits with validation
            splits = self.time_series_split(X, y, n_splits=n_splits)
            
            cv_scores = {
                'rmse': [],
                'r2': []
            }
            
            # Train model with validated splits
            for fold, (train_idx, val_idx) in enumerate(splits, 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = xgb.XGBRegressor(**model_params)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_val)
                cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
                cv_scores['r2'].append(r2_score(y_val, y_pred))
                
                logging.info(
                    f"Fold {fold} - "
                    f"Train: {X_train.index.min()} to {X_train.index.max()}, "
                    f"Val: {X_val.index.min()} to {X_val.index.max()}, "
                    f"RMSE: {cv_scores['rmse'][-1]:.4f}, "
                    f"R2: {cv_scores['r2'][-1]:.4f}"
                )
            
            # Train final model on all data
            final_model = xgb.XGBRegressor(**model_params)
            final_model.fit(X, y)
            
            metrics = {
                'mean_rmse': np.mean(cv_scores['rmse']),
                'std_rmse': np.std(cv_scores['rmse']),
                'mean_r2': np.mean(cv_scores['r2']),
                'std_r2': np.std(cv_scores['r2']),
                'training_period': {
                    'start': X.index.min().isoformat(),
                    'end': X.index.max().isoformat()
                }
            }
            
            return final_model, metrics
            
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise

    def save_model_and_metadata(self, model: xgb.XGBRegressor, 
                              feature_cols: List[str],
                              metrics: Dict,
                              model_name: Optional[str] = None) -> str:
        """Save model and associated metadata with enhanced temporal information"""
        try:
            if model_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"xgboost_timeseries_{timestamp}"
            
            # Save model
            model_path = os.path.join(self.model_save_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            
            # Save feature importance
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            importance_path = os.path.join(self.model_save_dir, f"{model_name}_feature_importance.csv")
            importance_df.to_csv(importance_path, index=False)
            
            # Save feature configuration
            feature_config = {
                'feature_names': feature_cols,
                'creation_time': datetime.now().isoformat(),
                'training_period': metrics.get('training_period', {}),
                'description': 'Feature configuration for price prediction model'
            }
            feature_config_path = os.path.join(self.model_save_dir, f"{model_name}_feature_config.json")
            with open(feature_config_path, 'w') as f:
                json.dump(feature_config, f, indent=4)
            
            # Save metrics
            metrics_path = os.path.join(self.model_save_dir, f"{model_name}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Save scaler
            if hasattr(self, 'scaler') and self.scaler is not None:
                scaler_path = os.path.join(self.model_save_dir, f"{model_name}_scaler.joblib")
                joblib.dump(self.scaler, scaler_path)
            
            logging.info(f"Model saved to: {model_path}")
            logging.info(f"Feature importance saved to: {importance_path}")
            logging.info(f"Feature configuration saved to: {feature_config_path}")
            logging.info(f"Metrics saved to: {metrics_path}")
            
            return model_path
            
        except Exception as e:
            logging.error(f"Error saving model and metadata: {e}")
            raise

    def train_and_save(self, table_name: str, 
                      target_col: str,
                      prediction_horizon: int = 1,
                      feature_cols: Optional[List[str]] = None,
                      model_params: Optional[Dict] = None,
                      model_name: Optional[str] = None) -> Tuple[str, Dict]:
        """Complete training pipeline with temporal validation"""
        try:
            # Load and preprocess data
            df = self.load_data_from_db(table_name)
            
            # Prepare features and target
            X, y = self.prepare_features_target(
                df, 
                target_col,
                feature_cols=feature_cols,
                prediction_horizon=prediction_horizon
            )
            
            # Train model
            model, metrics = self.train_model(X, y, model_params)
            
            # Save everything
            model_path = self.save_model_and_metadata(
                model,
                X.columns.tolist(),
                metrics,
                model_name=model_name
            )
            
            return model_path, metrics
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise