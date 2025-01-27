

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
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def setup_directories(self):
        os.makedirs(self.model_save_dir, exist_ok=True)
        





    def load_data_from_db(self, table_name: str) -> pd.DataFrame:
        """Load data from SQLite database with proper datetime and numeric handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"SELECT * FROM {table_name}"
            
            # Define numeric columns
            numeric_columns = [
                'Price', 'Equity', 'Balance', 'Profit', 'Positions', 
                'Score', 'ExitScore'
            ]
            
            # Read with numeric columns specified
            df = pd.read_sql_query(query, conn, coerce_float=True)
            
            # Convert date and time to datetime
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df = df.set_index('DateTime')
            df = df.sort_index()
            
            # Force numeric conversion for specific columns
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert any columns that should be numeric based on their content
            for col in df.columns:
                if col not in ['Date', 'Time', 'Symbol']:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        continue
            
            logging.info(f"Loaded {len(df)} rows from {table_name}")
            logging.info(f"Numeric columns: {df.select_dtypes(include=['float64', 'int64']).columns.tolist()}")
            
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
        """Prepare features and target with thorough feature handling"""
        try:
            # Validate inputs
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in DataFrame")
            
            # Get feature columns if not provided
            if feature_cols is None:
                feature_cols = [col for col in df.columns if col != target_col 
                            and col not in ['Date', 'Time', 'id', 'DateTime']]
            
            # Log feature information
            logging.info(f"Using features: {feature_cols}")
            
            # Check for missing features
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                logging.warning(f"Missing columns: {missing_cols}")
                for col in missing_cols:
                    df[col] = 0
            
            # Create target variable with future values
            y = df[target_col].shift(-prediction_horizon)
            y = y.ffill().bfill()  # Fill any NaN values
            
            # Select and order features
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
            X_scaled = X_scaled[:-prediction_horizon]
            y = y[:-prediction_horizon]
            
            # Log shapes for debugging
            logging.info(f"Feature shape: {X_scaled.shape}, Target shape: {y.shape}")
            
            return X_scaled, y
            
        except Exception as e:
            logging.error(f"Error in prepare_features_target: {e}")
            raise

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rich temporal features"""
        df = df.copy()
        
        # Time-based features
        df['Hour'] = df.index.hour
        df['DayOfWeek'] = df.index.dayofweek
        df['DayOfMonth'] = df.index.day
        df['WeekOfYear'] = df.index.isocalendar().week
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        
        # Session times (assuming market sessions)
        df['IsAsiaSession'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
        df['IsEuropeSession'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
        df['IsUSSession'] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(int)
        
        return df
        
    def create_lagged_features(self, df: pd.DataFrame, columns: List[str], 
                            lags: List[int]) -> pd.DataFrame:
        """Create lagged features for specified columns"""
        df = df.copy()
        
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
        return df
        



    def time_series_split(self, X: pd.DataFrame, y: pd.Series, 
                        n_splits: int = 5) -> TimeSeriesSplit:
        """Create time series cross-validation splits"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return tscv.split(X)
        


    def save_model_and_metadata(self, model: xgb.XGBRegressor, 
                            feature_cols: List[str],
                            metrics: Dict,
                            model_name: Optional[str] = None) -> str:
        """Save model and associated metadata with feature information"""
        try:
            if model_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"xgboost_timeseries_{timestamp}"
            
            # Save model using joblib
            model_path = os.path.join(self.model_save_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            
            # Save feature names and importance
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            importance_path = os.path.join(self.model_save_dir, f"{model_name}_feature_importance.csv")
            importance_df.to_csv(importance_path, index=False)
            
            # Save feature names separately for easier loading
            feature_names_path = os.path.join(self.model_save_dir, f"{model_name}_feature_names.json")
            with open(feature_names_path, 'w') as f:
                json.dump({
                    'feature_names': feature_cols,
                    'creation_time': datetime.now().isoformat()
                }, f, indent=4)
            
            # Save metrics
            metrics_path = os.path.join(self.model_save_dir, f"{model_name}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Save feature scaler if it exists
            if hasattr(self, 'scaler') and self.scaler is not None:
                scaler_path = os.path.join(self.model_save_dir, f"{model_name}_scaler.joblib")
                joblib.dump(self.scaler, scaler_path)
            
            logging.info(f"Model saved to: {model_path}")
            logging.info(f"Feature importance saved to: {importance_path}")
            logging.info(f"Feature names saved to: {feature_names_path}")
            logging.info(f"Metrics saved to: {metrics_path}")
            
            return model_path
            
        except Exception as e:
            logging.error(f"Error saving model and metadata: {e}")
            raise

    def train_and_save(self, table_name: str, 
                        target_col: str,
                        prediction_horizon: int = 1,
                        feature_params: Optional[Dict] = None,
                        feature_cols: Optional[List[str]] = None,
                        model_params: Optional[Dict] = None,
                        model_name: Optional[str] = None) -> Tuple[str, Dict]:
        """Complete training pipeline with temporal considerations"""
        try:
            # Load and preprocess data
            df = self.load_data_from_db(table_name)
            df = self.preprocess_data(df, feature_params)
            
            # Prepare features and target
            X, y = self.prepare_features_target(
                df, 
                target_col,
                feature_cols=feature_cols,  # Pass selected features
                prediction_horizon=prediction_horizon
            )
            
            # Train model
            model, metrics = self.train_model(X, y, model_params)
            
            # Save everything
            model_path = self.save_model_and_metadata(
                model,
                X.columns.tolist(),
                metrics,
                model_name=model_name  # Pass the model name to save_model_and_metadata
            )
            
            return model_path, metrics
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise

    def create_rolling_features(self, df: pd.DataFrame, columns: List[str], 
                            windows: List[int]) -> pd.DataFrame:
        """Create rolling statistics features efficiently"""
        try:
            df = df.copy()
            new_features = {}
            
            for col in columns:
                if col not in df.columns:
                    logging.warning(f"Column {col} not found for rolling features")
                    continue
                    
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    logging.warning(f"Could not convert {col} to numeric, skipping rolling features")
                    continue
                
                for window in windows:
                    try:
                        rolling = df[col].rolling(window=window, min_periods=1)
                        
                        new_features[f'{col}_rolling_mean_{window}'] = rolling.mean()
                        new_features[f'{col}_rolling_std_{window}'] = rolling.std()
                        new_features[f'{col}_rolling_min_{window}'] = rolling.min()
                        new_features[f'{col}_rolling_max_{window}'] = rolling.max()
                        new_features[f'{col}_momentum_{window}'] = df[col] - df[col].shift(window)
                        
                    except Exception as e:
                        logging.warning(f"Error creating rolling features for {col} with window {window}: {e}")
                        continue
            
            if new_features:
                new_features_df = pd.DataFrame(new_features, index=df.index)
                new_features_df = new_features_df.ffill().bfill()
                df = pd.concat([df, new_features_df], axis=1)
            
            return df
            
        except Exception as e:
            logging.error(f"Error in create_rolling_features: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame, feature_params: Dict = None) -> pd.DataFrame:
        """Preprocess data with temporal considerations and NaN handling"""
        try:
            # Fill NaN values
            df = df.ffill().bfill()
            
            # Convert boolean columns to int
            bool_columns = df.select_dtypes(include=['bool']).columns
            for col in bool_columns:
                df[col] = df[col].astype(int)
            
            # Create temporal features
            df = self.create_temporal_features(df)
            
            if feature_params is None:
                feature_params = {
                    'lag_columns': ['Price', 'Equity', 'Balance', 'Profit'],
                    'lag_values': [1, 5, 10, 20],
                    'rolling_columns': ['Price', 'Score', 'ExitScore'],
                    'rolling_windows': [5, 10, 20, 50]
                }
            
            all_feature_cols = (
                feature_params.get('lag_columns', []) + 
                feature_params.get('rolling_columns', [])
            )
            
            numeric_features = {}
            for col in all_feature_cols:
                if col in df.columns:
                    try:
                        numeric_features[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        logging.warning(f"Could not convert {col} to numeric")
            
            if numeric_features:
                df = df.assign(**numeric_features)
            
            # Create features
            if 'lag_columns' in feature_params and 'lag_values' in feature_params:
                df = self.create_lagged_features(
                    df, 
                    feature_params['lag_columns'],
                    feature_params['lag_values']
                )
            
            if 'rolling_columns' in feature_params and 'rolling_windows' in feature_params:
                df = self.create_rolling_features(
                    df,
                    feature_params['rolling_columns'],
                    feature_params['rolling_windows']
                )
            
            # Drop non-feature columns
            columns_to_drop = ['id', 'Date', 'Time', 'Symbol']
            df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
            
            # Fill any remaining NaN values
            df = df.fillna(0)
            
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            logging.info(f"Preprocessed data shape: {df.shape}")
            logging.info(f"Number of numeric columns: {len(numeric_cols)}")
            
            if df.isnull().any().any():
                nan_columns = df.columns[df.isnull().any()].tolist()
                logging.warning(f"NaN values found in columns: {nan_columns}")
                df = df.fillna(0)
            
            return df
            
        except Exception as e:
            logging.error(f"Error preprocessing data: {e}")
            raise



    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                    model_params: Optional[Dict] = None,
                    n_splits: int = 5) -> Tuple[xgb.XGBRegressor, Dict]:
        """Train model with time series cross-validation using basic parameters"""
        try:
            if model_params is None:
                model_params = self.model_params.copy()
            else:
                model_params = model_params.copy()
                
            # Remove any potentially problematic parameters
            for param in ['early_stopping_rounds', 'eval_metric']:
                model_params.pop(param, None)
            
            # Initialize metrics storage
            cv_scores = {
                'rmse': [],
                'r2': []
            }
            
            # Get time series splits
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            # Train model with cross-validation
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                model = xgb.XGBRegressor(**model_params)
                
                # Basic fit without additional parameters
                model.fit(X_train, y_train)
                
                # Evaluate fold
                y_pred = model.predict(X_val)
                cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
                cv_scores['r2'].append(r2_score(y_val, y_pred))
                
                logging.info(f"Fold {fold} - RMSE: {cv_scores['rmse'][-1]:.4f}, R2: {cv_scores['r2'][-1]:.4f}")
            
            # Train final model on all data
            final_model = xgb.XGBRegressor(**model_params)
            final_model.fit(X, y)
            
            # Calculate average metrics
            metrics = {
                'mean_rmse': np.mean(cv_scores['rmse']),
                'std_rmse': np.std(cv_scores['rmse']),
                'mean_r2': np.mean(cv_scores['r2']),
                'std_r2': np.std(cv_scores['r2'])
            }
            
            return final_model, metrics
            
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise



