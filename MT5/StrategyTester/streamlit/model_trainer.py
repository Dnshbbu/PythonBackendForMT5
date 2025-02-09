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
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from model_repository import ModelRepository
import torch
from torch.utils.data import Dataset, DataLoader
from model_implementations import TimeSeriesDataset, LSTMTimeSeriesModel
from mlflow_utils import MLflowManager

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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mlflow_manager = MLflowManager()

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
                              prediction_horizon: int = 1,
                              model_type: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target with support for LSTM sequences"""
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
        Train model with support for incremental learning using a consistent train–test split.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_params: Model parameters dictionary (should include 'model_type')
            existing_model: Optional existing model for incremental training
        
        Returns:
            Tuple of (model, metrics)
        """
        try:
            if not model_params:
                model_params = {}

            # Get model type from parameters; default to 'xgboost'
            model_type = model_params.get('model_type', 'xgboost')
            # Remove model_type from parameters as it's not a valid model parameter
            model_params = {k: v for k, v in model_params.items() if k != 'model_type'}

            # Perform a consistent train–test split for all models
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )

            # Log the time range for training and validation sets
            logging.info(f"Training data range: {X_train.index.min()} to {X_train.index.max()}")
            logging.info(f"Validation data range: {X_val.index.min()} to {X_val.index.max()}")

            if model_type == 'xgboost':
                from xgboost import XGBRegressor
                default_params = {
                    'objective': 'reg:squarederror',
                    'n_estimators': 1000,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 1
                }
                # Update default parameters with provided ones
                model_params = {**default_params, **model_params}

                if existing_model is not None and isinstance(existing_model, XGBRegressor):
                    logging.info("Performing incremental training with existing model")
                    current_n_trees = existing_model.n_estimators
                    n_additional_trees = 50
                    model_params['n_estimators'] = n_additional_trees

                    try:
                        existing_booster = existing_model.get_booster()
                    except Exception as e:
                        logging.warning(f"Could not get booster from existing model: {e}")
                        existing_booster = None

                    model = XGBRegressor(**model_params)
                    fit_params = {
                        'eval_set': [(X_val, y_val)],
                        'verbose': False
                    }
                    if existing_booster is not None:
                        fit_params['xgb_model'] = existing_booster
                    model.fit(X_train, y_train, **fit_params)
                    training_type = 'incremental'
                    total_trees = current_n_trees + n_additional_trees
                else:
                    logging.info("Performing full XGBoost training")
                    model = XGBRegressor(**model_params)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                    training_type = 'full'
                    total_trees = model_params['n_estimators']

            elif model_type == 'random_forest':
                from sklearn.ensemble import RandomForestRegressor
                logging.info("Training Random Forest model")
                default_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42
                }
                model_params = {**default_params, **model_params}
                model = RandomForestRegressor(**model_params)
                model.fit(X_train, y_train)
                training_type = 'full'
                total_trees = model_params['n_estimators']

            else:  # Decision Tree
                from sklearn.tree import DecisionTreeRegressor
                logging.info("Training Decision Tree model")
                # Filter only Decision Tree related parameters
                model_params = {k: v for k, v in model_params.items() 
                                if k in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state']}
                default_params = {
                    'max_depth': 6,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42
                }
                model_params = {**default_params, **model_params}
                model = DecisionTreeRegressor(**model_params)
                model.fit(X_train, y_train)
                training_type = 'full'
                total_trees = 1  # Decision tree represents a single tree

            # Calculate metrics on the validation set
            predictions = model.predict(X_val)
            mse = np.mean((y_val - predictions) ** 2)
            r2 = 1 - (np.sum((y_val - predictions) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))

            metrics = {
                'rmse': float(np.sqrt(mse)),
                'r2': float(r2),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'training_type': training_type,
                'training_period': {
                    'start': X.index.min().isoformat(),
                    'end': X.index.max().isoformat()
                },
                'model_type': model_type
            }

            # Add model-specific metrics
            if model_type in ['xgboost', 'random_forest']:
                metrics['n_trees'] = total_trees
                if model_type == 'xgboost' and hasattr(model, 'evals_result'):
                    validation_results = model.evals_result()
                    if validation_results:
                        metrics['validation_rmse'] = min(validation_results['validation_0']['rmse'])

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
            # Determine model type
            if isinstance(model, xgb.XGBRegressor):
                model_type = 'xgboost'
            elif isinstance(model, DecisionTreeRegressor):
                model_type = 'decision_tree'
            else:
                model_type = 'unknown'
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use consistent naming convention
            model_name = f"{model_type}_single_{timestamp}"
            
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
                                model_name: Optional[str] = None,
                                model_type: str = 'xgboost',
                                base_model_path: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Train and save a model using data from multiple tables
        
        Args:
            table_names: List of table names to use for training
            target_col: Target column name
            prediction_horizon: Number of steps ahead to predict
            feature_cols: List of feature column names
            model_params: Model parameters dictionary
            model_name: Optional name for the model
            model_type: Type of model to train ('xgboost', 'lstm', etc.)
            base_model_path: Optional path to base model for incremental training
            
        Returns:
            Tuple of (model_path, metrics)
        """
        try:
            # Generate or validate model name first
            if not model_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                training_type = 'multi' if len(table_names) > 1 else 'single'
                model_name = f"{model_type}_{training_type}_{timestamp}"
            else:
                # Ensure model_name includes model_type prefix if not already present
                if not model_name.startswith(f"{model_type}_"):
                    model_name = f"{model_type}_{model_name}"

            # Start MLflow run with the same model name
            with self.mlflow_manager.start_run(run_name=model_name):
                # Prepare MLflow parameters
                mlflow_params = {
                    'model_type': model_type,
                    'target_col': target_col,
                    'prediction_horizon': prediction_horizon,
                    'table_names': table_names,
                    'feature_cols': feature_cols,
                    'base_model_path': base_model_path
                }
                if model_params:
                    mlflow_params.update(model_params)
                
                # Log parameters
                self.mlflow_manager.log_params(mlflow_params)

                # Load and combine data from all tables
                df = self.load_data_from_multiple_tables(table_names)
                logging.info(f"Combined data shape: {df.shape}")
                
                # Prepare features and target
                X, y = self.prepare_features_target(
                    df, target_col, feature_cols or [], 
                    prediction_horizon, model_type
                )
                logging.info(f"Prepared features shape: {X.shape}, target shape: {y.shape}")
                
                # Load base model if provided
                existing_model = None
                if base_model_path and os.path.exists(base_model_path):
                    try:
                        if model_type == 'xgboost':
                            existing_model = joblib.load(base_model_path)
                            logging.info(f"Loaded base XGBoost model from {base_model_path}")
                        elif model_type == 'lstm':
                            existing_model = torch.load(base_model_path)
                            logging.info(f"Loaded base LSTM model from {base_model_path}")
                    except Exception as e:
                        logging.error(f"Error loading base model: {e}")
                        existing_model = None
                
                # Add model_type to model_params
                if model_params is None:
                    model_params = {}
                model_params['model_type'] = model_type

                # Train or update model
                if model_type == 'lstm':
                    # Create LSTM model instance
                    model = LSTMTimeSeriesModel()
                    model, metrics = model.train(
                        X=X,
                        y=y,
                        base_model=existing_model,
                        **model_params
                    )
                else:
                    # Train other models using existing logic
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
                
                # Save model based on type
                if model_type == 'lstm':
                    model_path = model.save(self.models_dir, model_name)
                else:
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
                
                # Store model information in repository using the same model name
                try:
                    model_repo = ModelRepository(self.db_path)
                    feature_importance = {}
                    
                    # Get feature importance if available
                    if model_type == 'lstm':
                        # LSTM uses equal feature importance
                        importance = 1.0 / len(X.columns)
                        feature_importance = {feat: importance for feat in X.columns}
                    elif hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(X.columns, model.feature_importances_))
                    elif model_type == 'xgboost':
                        importance_scores = model.get_booster().get_score(importance_type='gain')
                        feature_importance = {k: v for k, v in importance_scores.items()}
                    
                    model_repo.store_model_info(
                        model_name=model_name,  # Use the same model name consistently
                        model_type=model_type,
                        training_type='multi' if len(table_names) > 1 else 'single',
                        prediction_horizon=prediction_horizon,
                        features=X.columns.tolist(),
                        feature_importance=feature_importance,
                        model_params=model_params,
                        metrics=metrics,
                        training_tables=table_names,
                        training_period={
                            'start': str(X.index.min()),
                            'end': str(X.index.max())
                        },
                        data_points=len(X),
                        model_path=model_path,
                        scaler_path=os.path.join(self.models_dir, f"{model_name}_scaler.joblib")
                    )
                except Exception as repo_error:
                    logging.warning(f"Error storing model in repository: {repo_error}")
                
                # Log metrics to MLflow
                self.mlflow_manager.log_metrics(metrics)

                # Log model and artifacts using the same model name
                self.mlflow_manager.log_model(model, model_name)
                self.mlflow_manager.log_artifact(model_path)

                # If feature importance exists, log it
                if 'feature_importance' in metrics:
                    self.mlflow_manager.log_feature_importance(metrics['feature_importance'])
                
                return model_path, metrics
            
        except Exception as e:
            logging.error(f"Error in train_and_save_multi_table with MLflow: {e}")
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

        