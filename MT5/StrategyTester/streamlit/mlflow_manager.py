from typing import Dict, Any, Optional, Union, List
import mlflow
from datetime import datetime
import logging
import os

class MLflowManager:
    """Base class for MLflow management"""
    
    def __init__(self, tracking_uri: str, experiment_name: str):
        """Initialize MLflow manager
        
        Args:
            tracking_uri: URI for MLflow tracking server
            experiment_name: Name of the experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Set up MLflow tracking"""
        mlflow.set_tracking_uri(self.tracking_uri)
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
        mlflow.set_experiment(self.experiment_name)
    
    def _generate_run_id(self) -> str:
        """Generate a unique run ID in the format: run_YYYYMMDD_HHMMSS_fff"""
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]
        return f"run_{current_time}"
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run
        
        Args:
            run_name: Optional name for the run. If not provided, generates a unique run ID
            
        Returns:
            MLflow ActiveRun object
        """
        if run_name is None:
            run_name = self._generate_run_id()
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow
        
        Args:
            params: Dictionary of parameters to log
        """
        for param_name, param_value in params.items():
            if param_value is not None:  # Only log non-None parameters
                if isinstance(param_value, (list, tuple)):
                    # Convert lists/tuples to string representation
                    param_value = str(param_value)
                elif isinstance(param_value, dict):
                    # Convert dict to JSON string
                    import json
                    param_value = json.dumps(param_value)
                mlflow.log_param(param_name, param_value)
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]]):
        """Log metrics to MLflow
        
        Args:
            metrics: Dictionary of metrics to log
        """
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)
    
    def log_artifacts(self, local_dir: str):
        """Log artifacts to MLflow
        
        Args:
            local_dir: Local directory containing artifacts to log
        """
        mlflow.log_artifacts(local_dir)

class MLflowTrainingManager(MLflowManager):
    """MLflow manager for model training"""
    
    def __init__(self, tracking_uri: str):
        """Initialize training manager with 'trading_models' experiment"""
        super().__init__(tracking_uri, experiment_name="trading_models")
    
    def log_training_info(self, 
                         model_type: str,
                         model_name: str,
                         training_type: str,
                         features: List[str],
                         target: str,
                         training_params: Dict[str, Any],
                         metrics: Dict[str, Union[float, int]],
                         data_info: Dict[str, Any],
                         model_params: Optional[Dict[str, Any]] = None,
                         feature_importance: Optional[Dict[str, Any]] = None,
                         additional_metadata: Optional[Dict[str, Any]] = None):
        """Log comprehensive training information
        
        Args:
            model_type: Type of the model (e.g., 'ARIMA', 'Prophet', 'VAR')
            model_name: Name of the model
            training_type: Type of training (e.g., 'single', 'multi', 'incremental')
            features: List of feature names
            target: Target variable name
            training_params: Basic training parameters (n_lags, prediction_horizon, etc.)
            metrics: Training metrics
            data_info: Information about training data (tables, periods, data points)
            model_params: Model-specific parameters
            feature_importance: Feature importance information
            additional_metadata: Any additional metadata to log
        """
        logging.info(f"MLflowTrainingManager - Preparing to log parameters")
        logging.info(f"Target variable: {target}")
        
        # Basic information
        params = {
            'model_type': model_type,
            'model_name': model_name,
            'training_type': training_type,
            'features': features,
            'target_col': target,  # Changed to target_col as requested
            'training_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logging.info(f"Basic parameters prepared: {params}")
        
        # Add data information
        params.update({
            'table_names': data_info.get('table_names'),
            'data_points': data_info.get('data_points'),
            'training_period': data_info.get('training_period', {})
        })
        
        # Add training parameters
        params.update(training_params)
        
        # Add model-specific parameters if provided
        if model_params:
            # Handle different model types
            if model_type in ['ARIMA', 'SARIMA']:
                params.update({
                    'order': model_params.get('order'),
                    'seasonal_order': model_params.get('seasonal_order'),
                    'seasonal': model_params.get('seasonal'),
                    'seasonal_period': model_params.get('seasonal_period')
                })
            elif model_type == 'Auto ARIMA':
                params.update({
                    'max_p': model_params.get('max_p'),
                    'max_d': model_params.get('max_d'),
                    'max_q': model_params.get('max_q'),
                    'seasonal': model_params.get('seasonal'),
                    'seasonal_period': model_params.get('seasonal_period')
                })
            elif model_type == 'Prophet':
                params.update({
                    'changepoint_prior_scale': model_params.get('changepoint_prior_scale'),
                    'seasonality_prior_scale': model_params.get('seasonality_prior_scale')
                })
            elif model_type == 'VAR':
                params.update({
                    'maxlags': model_params.get('maxlags'),
                    'ic': model_params.get('ic'),
                    'trend': model_params.get('trend')
                })
        
        # Add feature importance if provided
        if feature_importance:
            params['feature_importance'] = feature_importance
        
        # Add additional metadata if provided
        if additional_metadata:
            # Add stationarity test results
            if 'stationarity_test' in additional_metadata:
                params.update({
                    'adf_statistic': additional_metadata['stationarity_test'].get('adf_statistic'),
                    'adf_pvalue': additional_metadata['stationarity_test'].get('adf_pvalue')
                })
            
            # Add model specification details
            if 'model_specification' in additional_metadata:
                params.update({
                    'trend_specification': additional_metadata['model_specification'].get('trend_specification'),
                    'has_seasonal': additional_metadata['model_specification'].get('has_seasonal'),
                    'model_seasonal_period': additional_metadata['model_specification'].get('seasonal_period')
                })
            
            # Add estimation details
            if 'estimation_details' in additional_metadata:
                params.update({
                    'nobs': additional_metadata['estimation_details'].get('nobs'),
                    'df_model': additional_metadata['estimation_details'].get('df_model'),
                    'df_resid': additional_metadata['estimation_details'].get('df_resid')
                })
            
            # Add preprocessing info for VAR models
            if 'preprocessing' in additional_metadata:
                params.update({
                    'diff_orders': additional_metadata['preprocessing'].get('diff_orders'),
                    'has_scaler': additional_metadata['preprocessing'].get('has_scaler')
                })
        
        # Log all parameters and metrics
        self.log_params(params)
        self.log_metrics(metrics)

class MLflowPredictionManager(MLflowManager):
    """MLflow manager for model predictions"""
    
    def __init__(self, tracking_uri: str):
        """Initialize prediction manager with 'model_predictions' experiment"""
        super().__init__(tracking_uri, experiment_name="model_predictions")
    
    def log_prediction_info(self,
                          model_name: str,
                          data_source: str,
                          prediction_params: Dict[str, Any],
                          metrics: Dict[str, Union[float, int]],
                          additional_params: Optional[Dict[str, Any]] = None):
        """Log prediction information
        
        Args:
            model_name: Name of the model used for prediction
            data_source: Source of the data (e.g., table name)
            prediction_params: All prediction parameters in a flat structure
            metrics: Prediction metrics
            additional_params: Additional parameters to log (optional)
        """
        # Log basic information
        params = {
            'model_name': model_name,
            'data_source': data_source,
            'prediction_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add all prediction parameters
        params.update(prediction_params)
        
        # Add any additional parameters if provided
        if additional_params:
            params.update(additional_params)
        
        # Log all parameters and metrics
        self.log_params(params)
        self.log_metrics(metrics) 