"""MLflow logging utility class for standardized logging"""

import mlflow
from datetime import datetime
from typing import Dict, Any, Optional
from mlflow_constants import (
    TRAINING_PARAMS, PREDICTION_PARAMS, MODEL_SPECIFIC_PARAMS,
    TRAINING_METRICS, PREDICTION_METRICS, EXPERIMENT_NAMES,
    RUN_NAME_FORMATS
)

class MLflowLogger:
    def __init__(self, mlflow_db_path: str):
        """Initialize MLflow logger
        
        Args:
            mlflow_db_path: Path to MLflow database
        """
        self.mlflow_db_path = mlflow_db_path
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
        
    def _setup_experiment(self, experiment_type: str) -> str:
        """Set up MLflow experiment
        
        Args:
            experiment_type: Type of experiment ('training' or 'prediction')
            
        Returns:
            Experiment ID
        """
        experiment_name = EXPERIMENT_NAMES[experiment_type]
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
            
        mlflow.set_experiment(experiment_name)
        return experiment_id
    
    def _generate_run_name(self, experiment_type: str, **kwargs) -> str:
        """Generate run name based on experiment type
        
        Args:
            experiment_type: Type of experiment ('training' or 'prediction')
            **kwargs: Additional parameters for run name formatting
            
        Returns:
            Generated run name
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if experiment_type == 'prediction':
            timestamp = f"{timestamp}_{str(datetime.now().microsecond)[:3]}"
            
        name_format = RUN_NAME_FORMATS[experiment_type]
        return name_format.format(timestamp=timestamp, **kwargs)
    
    def log_training(self, model_type: str, params: Dict[str, Any], metrics: Dict[str, float],
                    training_type: str = 'single', **kwargs) -> str:
        """Log training run to MLflow
        
        Args:
            model_type: Type of model being trained
            params: Training parameters
            metrics: Training metrics
            training_type: Type of training
            **kwargs: Additional parameters
            
        Returns:
            Run name
        """
        self._setup_experiment('training')
        run_name = self._generate_run_name('training', model_type=model_type, 
                                         training_type=training_type)
        
        with mlflow.start_run(run_name=run_name):
            # Log common parameters
            for param_name in TRAINING_PARAMS:
                if param_name in params:
                    mlflow.log_param(param_name, params[param_name])
            
            # Log model-specific parameters
            if model_type.upper() in MODEL_SPECIFIC_PARAMS:
                for param_name in MODEL_SPECIFIC_PARAMS[model_type.upper()]:
                    if param_name in params:
                        mlflow.log_param(param_name, params[param_name])
            
            # Log metrics
            for metric_name in TRAINING_METRICS:
                if metric_name in metrics:
                    mlflow.log_metric(metric_name, metrics[metric_name])
            
            # Log additional parameters and metrics
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
                else:
                    mlflow.log_param(key, value)
        
        return run_name
    
    def log_prediction(self, model_name: str, source_table: str, params: Dict[str, Any],
                      metrics: Dict[str, float], **kwargs) -> str:
        """Log prediction run to MLflow
        
        Args:
            model_name: Name of the model used for prediction
            source_table: Source table for prediction
            params: Prediction parameters
            metrics: Prediction metrics
            **kwargs: Additional parameters
            
        Returns:
            Run name (run_id)
        """
        self._setup_experiment('prediction')
        run_name = self._generate_run_name('prediction')
        
        with mlflow.start_run(run_name=run_name):
            # Log standard parameters
            mlflow.log_param('model_name', model_name)
            mlflow.log_param('source_table', source_table)
            
            # Log other prediction parameters
            for param_name in PREDICTION_PARAMS:
                if param_name in params:
                    mlflow.log_param(param_name, params[param_name])
            
            # Log metrics
            for metric_name in PREDICTION_METRICS:
                if metric_name in metrics:
                    mlflow.log_metric(metric_name, metrics[metric_name])
            
            # Log additional parameters and metrics
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
                else:
                    mlflow.log_param(key, value)
        
        return run_name
    
    def get_run_info(self, run_name: str) -> Optional[Dict[str, Any]]:
        """Get run information from MLflow
        
        Args:
            run_name: Name of the run
            
        Returns:
            Dictionary containing run information or None if not found
        """
        client = mlflow.tracking.MlflowClient()
        for experiment_name in EXPERIMENT_NAMES.values():
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"tags.mlflow.runName = '{run_name}'"
                )
                if runs:
                    run = runs[0]
                    return {
                        'run_id': run.info.run_id,
                        'experiment_name': experiment_name,
                        'status': run.info.status,
                        'start_time': datetime.fromtimestamp(run.info.start_time/1000.0),
                        'end_time': datetime.fromtimestamp(run.info.end_time/1000.0) if run.info.end_time else None,
                        'params': run.data.params,
                        'metrics': run.data.metrics
                    }
        return None 