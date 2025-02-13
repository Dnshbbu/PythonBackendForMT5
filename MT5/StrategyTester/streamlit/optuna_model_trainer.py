import optuna
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso, HuberRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from pycaret_model_trainer import PyCaretModelTrainer

class OptunaModelTrainer(PyCaretModelTrainer):
    def __init__(self, db_path: str, models_dir: str):
        """Initialize with parent class initialization"""
        super().__init__(db_path, models_dir)
        
    def optimize_lightgbm(self, trial: optuna.Trial, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                         y_train: pd.Series, y_test: pd.Series) -> Tuple[float, Dict, Any]:
        """Optimize LightGBM hyperparameters"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'min_child_weight': trial.suggest_float('min_child_weight', 1, 5),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 30),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': 42,
            'verbose': -1,  # Suppress warnings
            'min_split_gain': 0.0,  # Allow splits with small gains
        }
        
        # Create evaluation dataset
        eval_set = [(X_test, y_test)]
        
        model = LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_metric='mae',
            eval_set=eval_set,
            callbacks=[
                optuna.integration.LightGBMPruningCallback(trial, 'mae')
            ]
        )
        
        y_pred = model.predict(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, 'lightgbm')
        # Adjust the optimization objective to balance MAE and Directional Accuracy
        score = metrics['MAE'] - 0.05 * metrics['DirectionalAccuracy']  # Lower is better
        
        return score, metrics, model

    def optimize_xgboost(self, trial: optuna.Trial, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        y_train: pd.Series, y_test: pd.Series) -> Tuple[float, Dict, Any]:
        """Optimize XGBoost hyperparameters"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'min_child_weight': trial.suggest_float('min_child_weight', 1, 5),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'alpha': trial.suggest_float('alpha', 0.0, 1.0),
            'lambda': trial.suggest_float('lambda', 0.0, 1.0),
            'random_state': 42,
            'verbosity': 0,  # Suppress warnings
            'objective': 'reg:squarederror'
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, 'xgboost')
        # Adjust the optimization objective to balance MAE and Directional Accuracy
        score = metrics['MAE'] - 0.05 * metrics['DirectionalAccuracy']  # Lower is better
        
        return score, metrics, model

    def optimize_random_forest(self, trial: optuna.Trial, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                             y_train: pd.Series, y_test: pd.Series) -> Tuple[float, Dict, Any]:
        """Optimize Random Forest hyperparameters"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_float('max_features', 0.3, 1.0),
            'random_state': 42,
            'n_jobs': -1  # Use all CPU cores
        }
        
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, 'random_forest')
        score = metrics['MAE'] - 0.05 * metrics['DirectionalAccuracy']  # Lower is better
        
        return score, metrics, model

    def optimize_gradient_boosting(self, trial: optuna.Trial, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                 y_train: pd.Series, y_test: pd.Series) -> Tuple[float, Dict, Any]:
        """Optimize Gradient Boosting hyperparameters"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'random_state': 42
        }
        
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, 'gradient_boosting')
        score = metrics['MAE'] - 0.05 * metrics['DirectionalAccuracy']
        
        return score, metrics, model

    def optimize_elastic_net(self, trial: optuna.Trial, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                           y_train: pd.Series, y_test: pd.Series) -> Tuple[float, Dict, Any]:
        """Optimize ElasticNet hyperparameters"""
        params = {
            'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
            'max_iter': 2000,
            'random_state': 42
        }
        
        model = ElasticNet(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, 'elastic_net')
        score = metrics['MAE'] - 0.05 * metrics['DirectionalAccuracy']
        
        return score, metrics, model

    def optimize_svr(self, trial: optuna.Trial, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                    y_train: pd.Series, y_test: pd.Series) -> Tuple[float, Dict, Any]:
        """Optimize Support Vector Regression hyperparameters"""
        params = {
            'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
            'epsilon': trial.suggest_float('epsilon', 1e-3, 1.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-3, 1.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly'])
        }
        
        model = SVR(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, 'svr')
        score = metrics['MAE'] - 0.05 * metrics['DirectionalAccuracy']
        
        return score, metrics, model

    def optimize_knn(self, trial: optuna.Trial, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                    y_train: pd.Series, y_test: pd.Series) -> Tuple[float, Dict, Any]:
        """Optimize K-Nearest Neighbors hyperparameters"""
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'p': trial.suggest_int('p', 1, 2),  # 1 for manhattan, 2 for euclidean
            'leaf_size': trial.suggest_int('leaf_size', 10, 50)
        }
        
        model = KNeighborsRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, 'knn')
        score = metrics['MAE'] - 0.05 * metrics['DirectionalAccuracy']
        
        return score, metrics, model

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, model_type: str) -> Dict:
        """Calculate all metrics for model evaluation"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        dir_acc = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
        
        metrics = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2),
            'MAPE': float(mape),
            'DirectionalAccuracy': float(dir_acc),
            'Model': model_type
        }
        return metrics

    def train_with_optuna(self, X: pd.DataFrame, y: pd.Series, 
                         model_type: str = 'lightgbm',
                         n_trials: int = 100,
                         timeout: Optional[int] = None) -> Tuple[Any, Dict]:
        """
        Train model using Optuna for hyperparameter optimization
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_type: Type of model to optimize ('lightgbm', 'xgboost', or 'random_forest')
            n_trials: Number of optimization trials
            timeout: Optional timeout in seconds
            
        Returns:
            Tuple of (best model, metrics dictionary)
        """
        # Convert data types to reduce memory usage
        X = X.astype('float32')
        y = y.astype('float32')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to DataFrame to maintain column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # Map model types to optimization functions
        optimize_funcs = {
            'lightgbm': self.optimize_lightgbm,
            'xgboost': self.optimize_xgboost,
            'random_forest': self.optimize_random_forest,
            'gradient_boosting': self.optimize_gradient_boosting,
            'elastic_net': self.optimize_elastic_net,
            'svr': self.optimize_svr,
            'knn': self.optimize_knn
        }
        
        if model_type not in optimize_funcs:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(optimize_funcs.keys())}")
        
        optimize_func = optimize_funcs[model_type]
        
        # Create Optuna study with pruning
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
        study = optuna.create_study(
            direction='minimize',
            pruner=pruner
        )
        
        # Store best metrics and model
        best_metrics = None
        best_model = None
        
        def objective(trial):
            nonlocal best_metrics, best_model
            score, metrics, model = optimize_func(trial, X_train_scaled, X_test_scaled, y_train, y_test)
            
            # Update best metrics and model if this trial has the best score
            if best_metrics is None or score < trial.study.best_value:
                best_metrics = metrics
                best_model = model
            
            return score
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Log best parameters and metrics
        logging.info(f"\nBest {model_type} parameters: {study.best_params}")
        logging.info(f"Best metrics: {best_metrics}")
        
        # Add study statistics to metrics
        best_metrics['optimization_history'] = {
            'values': study.trials_dataframe()['value'].tolist(),
            'params': [t.params for t in study.trials]
        }
        
        # Create final model dictionary
        final_model = {
            'model': best_model,
            'scaler': scaler,
            'feature_names': list(X.columns),
            'best_params': study.best_params
        }
        
        return final_model, best_metrics 