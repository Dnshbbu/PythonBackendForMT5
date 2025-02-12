import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import sqlite3
import logging
import joblib
import os
import json
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from pycaret.regression import *
from model_repository import ModelRepository
from mlflow_utils import MLflowManager

class PyCaretModelTrainer:
    def __init__(self, db_path: str, models_dir: str):
        """
        Initialize the PyCaret trainer with database and model paths
        
        Args:
            db_path: Path to SQLite database
            models_dir: Directory to save trained models
        """
        # Update the db_path to point to the logs directory
        if not os.path.exists(db_path) and os.path.exists(os.path.join('logs', db_path)):
            self.db_path = os.path.join('logs', db_path)
        else:
            self.db_path = db_path
            
        self.models_dir = models_dir
        self.setup_logging()
        self.setup_directories()
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
        
    def load_data_from_db(self, table_name: str) -> pd.DataFrame:
        """Load data from SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"""
                SELECT * FROM {table_name}
                ORDER BY Date, Time
            """
            
            df = pd.read_sql_query(query, conn, coerce_float=True)
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df = df.set_index('DateTime')
            
            logging.info(f"Loaded {len(df)} rows from {table_name}")
            return df
        finally:
            if conn:
                conn.close()

    def prepare_features_target(self, df: pd.DataFrame, 
                              target_col: str,
                              feature_cols: List[str],
                              prediction_horizon: int = 1,
                              n_lags: int = 3,
                              use_price_features: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            feature_cols: List of feature column names
            prediction_horizon: Number of steps ahead to predict
            n_lags: Number of previous price values to include as features
            use_price_features: Whether to use current and lagged price values as features
            
        Returns:
            Tuple of features DataFrame and target Series
        """
        # Create a copy to avoid modifying original data
        data = df.copy()
        features = feature_cols.copy()
        
        if use_price_features:
            # Add lagged price values as features
            for i in range(1, n_lags + 1):
                lag_col = f"{target_col}_lag_{i}"
                data[lag_col] = data[target_col].shift(i)
                features.append(lag_col)
            
            # Add current price as a feature
            features.append(target_col)
        
        # Shift target column up by prediction_horizon to align features with future target
        y = data[target_col].shift(-prediction_horizon)
        X = data[features]
        
        # Remove rows with NaN values created by shift operations
        if use_price_features:
            X = X.iloc[n_lags:]
            y = y.iloc[n_lags:]
        
        # Remove rows with NaN values from prediction horizon
        X = X[:-prediction_horizon]
        y = y[:-prediction_horizon]
        
        return X, y

    def train_pycaret_model(self, X: pd.DataFrame, y: pd.Series, 
                           model_params: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """
        Train models individually and select the best one
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_params: Optional parameters for setup and training
            
        Returns:
            Tuple of (trained model, metrics dictionary)
        """
        try:
            # Convert data types to reduce memory usage
            X = X.astype('float32')
            y = y.astype('float32')
            
            # Initialize models
            from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge
            from lightgbm import LGBMRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from xgboost import XGBRegressor
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.svm import SVR

            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Initialize models with parameters
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(alpha=1.0),
                'Lasso': Lasso(alpha=1.0),
                'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
                'LightGBM': LGBMRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=8,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=2,
                    random_state=42,
                    verbose=-1
                ),
                'XGBoost': XGBRegressor(
                    max_depth=8,
                    learning_rate=0.05,
                    n_estimators=1000,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=2,
                    objective='reg:squarederror',
                    random_state=42
                ),
                'Random Forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42
                ),
                'K Neighbors Regressor': KNeighborsRegressor(
                    n_neighbors=5,
                    weights='uniform',
                    algorithm='auto'
                ),
                'AdaBoost': AdaBoostRegressor(
                    n_estimators=100,
                    learning_rate=1.0,
                    random_state=42
                ),
                'Gradient Boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    subsample=1.0,
                    random_state=42
                ),
                'Support Vector Regression': SVR(
                    kernel='rbf',
                    C=1.0,
                    epsilon=0.1
                ),
                'Huber Regressor': HuberRegressor(
                    epsilon=1.35,
                    alpha=0.0001,
                    max_iter=100
                ),
                'Bayesian Ridge': BayesianRidge(
                    n_iter=300,
                    alpha_1=1e-6,
                    alpha_2=1e-6
                )
            }

            # Try to add CatBoost if available
            try:
                from catboost import CatBoostRegressor
                models['CatBoost'] = CatBoostRegressor(
                    iterations=1000,
                    learning_rate=0.05,
                    depth=6,
                    l2_leaf_reg=3,
                    verbose=False,
                    random_state=42
                )
            except ImportError:
                logging.warning("CatBoost is not installed. Skipping CatBoost model.")
            
            # Update model parameters if provided
            if model_params:
                model_type = model_params.get('model_type')
                logging.info(f"Requested model type: {model_type}")
                
                if model_type == "automl":
                    # Handle AutoML with selected models
                    selected_models = model_params.get('selected_models')
                    model_specific_params = model_params.get('model_specific_params', {})
                    
                    if selected_models:
                        logging.info(f"AutoML mode with selected models: {selected_models}")
                        # Filter to keep only selected models
                        filtered_models = {}
                        for name in selected_models:
                            if name in models:
                                filtered_models[name] = models[name]
                                # Apply model-specific parameters if available
                                if name in model_specific_params:
                                    logging.info(f"Applying parameters for {name}: {model_specific_params[name]}")
                                    filtered_models[name].set_params(**model_specific_params[name])
                            else:
                                logging.warning(f"Model {name} not found in available models")
                        models = filtered_models
                        if not models:
                            raise ValueError("None of the selected models are available")
                        logging.info(f"Using models: {list(models.keys())}")
                    else:
                        logging.info("AutoML mode with all available models")
                
                elif model_type in models:
                    logging.info(f"Training single model: {model_type}")
                    # If specific model parameters are provided, update them
                    model_specific_params = {k: v for k, v in model_params.items() if k != 'model_type' and k != 'cv' and k != 'selected_models'}
                    if model_specific_params:
                        logging.info(f"Applying model parameters: {model_specific_params}")
                        models[model_type].set_params(**model_specific_params)
                    # Use only the specified model
                    models = {model_type: models[model_type]}
                else:
                    logging.warning(f"Model type {model_type} not found in available models: {list(models.keys())}")
            
            best_score = float('inf')
            best_model = None
            best_name = None
            best_metrics = None
            all_metrics = {}  # Store metrics for all models
            
            def calculate_mape(y_true, y_pred):
                """Calculate Mean Absolute Percentage Error"""
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            def calculate_directional_accuracy(y_true, y_pred):
                """Calculate directional accuracy"""
                y_true_dir = np.sign(np.diff(y_true))
                y_pred_dir = np.sign(np.diff(y_pred))
                return np.mean(y_true_dir == y_pred_dir) * 100
            
            # Train and evaluate each model
            for name, model in models.items():
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate multiple metrics
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    mape = calculate_mape(y_test, y_pred)
                    dir_acc = calculate_directional_accuracy(y_test, y_pred)
                    
                    # Log all metrics
                    logging.info(f"\n{name} Performance Metrics:")
                    logging.info(f"MAE: {mae:.4f}")
                    logging.info(f"RMSE: {rmse:.4f}")
                    logging.info(f"R²: {r2:.4f}")
                    logging.info(f"MAPE: {mape:.2f}%")
                    logging.info(f"Directional Accuracy: {dir_acc:.2f}%")
                    
                    # Store metrics for this model
                    current_metrics = {
                        'MAE': float(mae),
                        'RMSE': float(rmse),
                        'R2': float(r2),
                        'MAPE': float(mape),
                        'DirectionalAccuracy': float(dir_acc)
                    }
                    
                    # Add model-specific metrics and data
                    if name in ['Random Forest', 'XGBoost', 'LightGBM', 'CatBoost']:
                        if hasattr(model, 'feature_importances_'):
                            current_metrics['feature_importance_detailed'] = {
                                'feature': list(X.columns),
                                'importance': model.feature_importances_.tolist()
                            }
                    
                    elif name in ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet']:
                        if hasattr(model, 'coef_'):
                            current_metrics['coefficients'] = {
                                'feature': list(X.columns),
                                'coefficient': model.coef_.tolist()
                            }
                    
                    elif name == 'K Neighbors Regressor':
                        # Calculate distances to nearest neighbors
                        distances, _ = model.kneighbors(X_test_scaled)
                        current_metrics['neighbor_distances'] = distances.flatten().tolist()
                    
                    elif name == 'Support Vector Regression':
                        if hasattr(model, 'support_vectors_'):
                            # Get first two components for visualization
                            sv_transformed = model.support_vectors_[:, :2] if model.support_vectors_.shape[1] > 1 else np.column_stack((model.support_vectors_, np.zeros_like(model.support_vectors_)))
                            current_metrics['support_vectors'] = {
                                'x': sv_transformed[:, 0].tolist(),
                                'y': sv_transformed[:, 1].tolist()
                            }
                    
                    # Store actual and predicted values for visualization
                    current_metrics['y_true'] = y_test.tolist()
                    current_metrics['y_pred'] = y_pred.tolist()
                    
                    # Store metrics for all models
                    all_metrics[name] = current_metrics
                    
                    # Still use MAE as the primary metric for model selection
                    if mae < best_score:
                        best_score = mae
                        best_model = model
                        best_name = name
                        best_metrics = current_metrics
                        
                except Exception as e:
                    logging.warning(f"Error training {name}: {str(e)}")
                    logging.warning("Full traceback:", exc_info=True)
                    continue
            
            if best_model is None:
                raise ValueError("No models were successfully trained")
            
            # Create metrics dictionary with all metrics
            metrics = {
                'Model': best_name,
                'Features': list(X.columns),
                'AllModels': all_metrics,  # Now includes model-specific visualizations
                **best_metrics  # Include all metrics from best model
            }
            
            # Save scaler with the model
            final_model = {
                'model': best_model,
                'scaler': scaler,
                'feature_names': list(X.columns)
            }
            
            return final_model, metrics
            
        except Exception as e:
            logging.error(f"Error in train_model: {str(e)}")
            raise

    def save_model_and_metadata(self, model: Dict, 
                              metrics: Dict,
                              model_name: Optional[str] = None) -> str:
        """
        Save trained model and its metadata
        
        Args:
            model: Dictionary containing model, scaler, and feature names
            metrics: Dictionary of model metrics
            model_name: Optional name for the model
            
        Returns:
            Path where model was saved
        """
        if model_name is None:
            model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        model_dir = os.path.join(self.models_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model components
        model_path = os.path.join(model_dir, "model.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        
        joblib.dump(model['model'], model_path)
        joblib.dump(model['scaler'], scaler_path)
        
        # Save metadata
        metadata = {
            'feature_names': model['feature_names'],
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'model_type': metrics['Model']
        }
        
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        return model_dir

    def train_and_save(self, 
                      table_names: List[str],
                      target_col: str,
                      prediction_horizon: int = 1,
                      feature_cols: Optional[List[str]] = None,
                      model_params: Optional[Dict] = None,
                      model_name: Optional[str] = None,
                      n_lags: int = 3,
                      use_price_features: bool = True) -> Tuple[str, Dict]:
        """
        Complete training pipeline: load data, train model, and save
        
        Args:
            table_names: List of table names to load data from
            target_col: Name of target column
            prediction_horizon: Number of steps ahead to predict
            feature_cols: Optional list of feature columns
            model_params: Optional model parameters
            model_name: Optional name for the model
            n_lags: Number of previous price values to include as features (default: 3)
            use_price_features: Whether to use current and previous prices as features (default: True)
            
        Returns:
            Tuple of (model directory path, metrics dictionary)
        """
        # Load and combine data from all tables
        dfs = []
        for table in table_names:
            df = self.load_data_from_db(table)
            dfs.append(df)
        combined_df = pd.concat(dfs)
        # Sort the combined DataFrame by DateTime index to maintain temporal consistency
        combined_df = combined_df.sort_index()
        logging.info(f"Combined data from {len(table_names)} tables, total rows: {len(combined_df)}")
        
        # Use all numeric columns as features if not specified
        if feature_cols is None:
            feature_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols if col != target_col]
        
        # Prepare features and target
        X, y = self.prepare_features_target(
            combined_df, target_col, feature_cols, prediction_horizon,
            n_lags=n_lags if use_price_features else 0,
            use_price_features=use_price_features
        )
        
        # Train model
        model, metrics = self.train_pycaret_model(X, y, model_params)
        
        # Save model and metadata
        model_dir = self.save_model_and_metadata(
            model, metrics, model_name
        )
        
        # Log to MLflow
        try:
            import mlflow
            with mlflow.start_run(run_name=model_name or "pycaret_model"):
                # Log metrics for all models
                for model_name, model_metrics in metrics['AllModels'].items():
                    # Only log numeric metrics
                    numeric_metrics = {
                        k: v for k, v in model_metrics.items() 
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    }
                    for metric_name, value in numeric_metrics.items():
                        try:
                            mlflow.log_metric(f"{model_name}_{metric_name}", float(value))
                        except (TypeError, ValueError) as e:
                            logging.warning(f"Could not log metric {metric_name} with value {value}: {str(e)}")
                
                # Log best model metrics separately
                best_metrics = {
                    "best_MAE": metrics.get("MAE"),
                    "best_RMSE": metrics.get("RMSE"),
                    "best_R2": metrics.get("R2"),
                    "best_MAPE": metrics.get("MAPE"),
                    "best_DirectionalAccuracy": metrics.get("DirectionalAccuracy")
                }
                
                # Log only valid numeric metrics
                for metric_name, value in best_metrics.items():
                    if value is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
                        try:
                            mlflow.log_metric(metric_name, float(value))
                        except (TypeError, ValueError) as e:
                            logging.warning(f"Could not log metric {metric_name} with value {value}: {str(e)}")
                
                # Log parameters
                params_to_log = {
                    "best_model_type": metrics.get("Model", ""),
                    "feature_columns": ", ".join(metrics.get("Features", [])),
                    "target_column": target_col,
                    "prediction_horizon": prediction_horizon,
                    "train_test_split": 0.2,
                    "shuffle": False,
                    "scaling": "StandardScaler",
                    "n_lags": n_lags if use_price_features else 0,
                    "use_price_features": use_price_features
                }
                
                # Log feature importance if available
                if "FeatureImportance" in metrics:
                    for feature, importance in metrics["FeatureImportance"].items():
                        if isinstance(importance, (int, float)):
                            params_to_log[f"importance_{feature}"] = float(importance)
                
                mlflow.log_params(params_to_log)
                
                # Log model artifacts
                if "model" in metrics:
                    mlflow.sklearn.log_model(metrics["model"], "model")
                
        except Exception as e:
            logging.warning(f"Error logging to MLflow: {str(e)}")
            logging.warning("Full traceback:", exc_info=True)
            
        return model_dir, metrics 