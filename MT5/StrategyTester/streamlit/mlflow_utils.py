import mlflow
import os
from typing import Dict, Any, Optional
import logging
from pathlib import Path
from datetime import datetime

class MLflowManager:
    def __init__(self, tracking_uri: str = None):
        """
        Initialize MLflow manager

        Args:
            tracking_uri: SQLite URI for MLflow tracking. 
                          If None, defaults to 'sqlite:///mlflow.db'
        """
        if tracking_uri is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(current_dir, 'mlflow.db')
            tracking_uri = f'sqlite:///{db_path}'

        mlflow.set_tracking_uri(tracking_uri)
        base_experiment_name = "trading_models"
        
        # Create the artifact directory and convert its path to a proper file URI
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.artifact_location = os.path.abspath(os.path.join(current_dir, 'mlflow_artifacts'))
        os.makedirs(self.artifact_location, exist_ok=True)
        artifact_uri = Path(self.artifact_location).absolute().as_uri()

        # Check if an experiment with the base name exists.
        existing_exp = mlflow.get_experiment_by_name(base_experiment_name)
        if existing_exp:
            # If the artifact location is not properly formatted, create a new experiment.
            if not existing_exp.artifact_location.startswith("file://"):
                unique_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.experiment_name = f"{base_experiment_name}_{unique_suffix}"
                self.experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=artifact_uri
                )
                logging.info(f"Existing experiment had invalid artifact URI. Created new experiment: {self.experiment_name}")
            else:
                self.experiment_name = base_experiment_name
                self.experiment_id = existing_exp.experiment_id
                logging.info(f"Using existing experiment: {self.experiment_name}")
        else:
            self.experiment_name = base_experiment_name
            self.experiment_id = mlflow.create_experiment(
                self.experiment_name,
                artifact_location=artifact_uri
            )
            logging.info(f"Created new experiment: {self.experiment_name}")
            
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name: Optional[str] = None) -> Any:
        """Start a new MLflow run"""
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow"""
        try:
            for key, value in params.items():
                if isinstance(value, (list, dict)):
                    # Convert lists/dicts to strings for MLflow
                    mlflow.log_param(key, str(value))
                else:
                    mlflow.log_param(key, value)
        except Exception as e:
            logging.warning(f"Error logging parameters to MLflow: {e}")

    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to MLflow"""
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
        except Exception as e:
            logging.warning(f"Error logging metrics to MLflow: {e}")

    def log_model(self, model: Any, model_name: str):
        """Log model to MLflow"""
        try:
            if hasattr(model, 'get_params'):  # scikit-learn compatible model
                mlflow.sklearn.log_model(model, model_name)
            else:
                # For custom models, save as artifact and log the file.
                import joblib
                temp_path = f"temp_{model_name}.joblib"
                joblib.dump(model, temp_path)
                self.log_artifact(temp_path)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            logging.warning(f"Error logging model to MLflow: {e}")

    def log_artifact(self, local_path: str):
        """Log an artifact file to MLflow"""
        try:
            if os.path.exists(local_path):
                mlflow.log_artifact(local_path)
            else:
                logging.warning(f"Artifact file not found: {local_path}")
        except Exception as e:
            logging.warning(f"Error logging artifact to MLflow: {e}")

    def log_feature_importance(self, feature_importance: Dict[str, float]):
        """Log feature importance as a parameter"""
        try:
            # Convert to string representation for MLflow
            mlflow.log_param("feature_importance", str(feature_importance))
        except Exception as e:
            logging.warning(f"Error logging feature importance to MLflow: {e}")

    def end_run(self):
        """End the current MLflow run"""
        try:
            mlflow.end_run()
        except Exception as e:
            logging.warning(f"Error ending MLflow run: {e}")