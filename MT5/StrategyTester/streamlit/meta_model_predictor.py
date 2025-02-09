import os
import logging
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import mlflow
import json
import torch
import xgboost as xgb
from typing import Dict, List, Optional, Tuple
from model_predictor import ModelPredictor
from run_predictions import HistoricalPredictor
from model_repository import ModelRepository

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_model_params(model_type: str) -> Dict:
    """Get model parameters based on model type"""
    params = {
        'xgboost': {
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 2,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
    }
    return params.get(model_type, {})

def generate_model_name(model_type: str, training_type: str, timestamp: Optional[str] = None) -> str:
    """Generate consistent model name"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_type}_{training_type}_{timestamp}"

class MetaModelPredictor:
    def __init__(self, db_path: str, models_dir: str, model_name: str):
        """
        Initialize the meta model predictor
        
        Args:
            db_path: Path to the database
            models_dir: Directory containing models
            model_name: Name of the meta model to use
        """
        self.db_path = db_path
        self.models_dir = models_dir
        self.model_name = model_name
        self.model_repository = ModelRepository(db_path)
        
        # Load meta model info
        self.meta_model_info = self.model_repository.get_meta_model_info(model_name)
        
        # Initialize base model predictors
        self.base_predictors = {}
        self.setup_base_predictors()
        
        self.setup_logging()

    def setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def setup_base_predictors(self):
        """Setup predictors for each base model"""
        try:
            # Get base model names from additional_metadata
            base_model_names = self.meta_model_info.get('base_model_names', [])
            if not base_model_names:
                raise ValueError("No base model names found in meta model metadata")
            
            logging.info(f"Setting up predictors for base models: {base_model_names}")
            
            # Initialize predictor for each base model
            for model_name in base_model_names:
                self.base_predictors[model_name] = HistoricalPredictor(
                    self.db_path,
                    self.models_dir,
                    model_name
                )
                
            logging.info(f"Successfully set up {len(self.base_predictors)} base predictors")
            
        except Exception as e:
            logging.error(f"Error setting up base predictors: {e}")
            raise

    def prepare_meta_features(self, base_predictions: List[Dict]) -> pd.DataFrame:
        """
        Prepare features for meta model from base model predictions
        
        Args:
            base_predictions: List of prediction results from base models
            
        Returns:
            DataFrame with features prepared for meta model
        """
        try:
            # Create a mapping of model names to generic model numbers
            model_names = sorted(self.base_predictors.keys())
            model_to_generic = {name: f'm{i+1}' for i, name in enumerate(model_names)}
            
            # Initialize DataFrame with datetime
            meta_features = pd.DataFrame()
            
            # Add features from each base model
            for model_name in model_names:
                generic_name = model_to_generic[model_name]
                model_preds = base_predictions[model_name]
                
                # Add predicted price
                meta_features[f'{generic_name}_price_pred'] = model_preds['Predicted_Price']
                
                # Add error
                meta_features[f'{generic_name}_error'] = model_preds['Error']
                
                # Add predicted change
                meta_features[f'{generic_name}_change_pred'] = model_preds['Predicted_Change']
            
            # Add common features
            meta_features['price_change'] = base_predictions[model_names[0]]['Price_Change']
            meta_features['price_volatility'] = base_predictions[model_names[0]]['Price_Volatility']
            
            # Ensure feature order matches training
            expected_features = self.meta_model_info['features']
            meta_features = meta_features[expected_features]
            
            return meta_features
            
        except Exception as e:
            logging.error(f"Error preparing meta features: {e}")
            raise

    def run_predictions(self, table_name: str) -> pd.DataFrame:
        """
        Run predictions using meta model
        
        Args:
            table_name: Name of the table to run predictions on
            
        Returns:
            DataFrame containing predictions and metrics
        """
        try:
            # Set MLflow tracking URI
            current_dir = os.path.dirname(os.path.abspath(__file__))
            mlflow_db = os.path.join(current_dir, "mlflow.db")
            mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
            
            # Set up MLflow experiment for predictions
            experiment = mlflow.get_experiment_by_name("model_predictions")
            if experiment is None:
                experiment_id = mlflow.create_experiment("model_predictions")
                logging.info(f"Created new MLflow experiment with ID: {experiment_id}")
            else:
                logging.info(f"Using existing MLflow experiment with ID: {experiment.experiment_id}")
            
            mlflow.set_experiment("model_predictions")
            
            # Create run name
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]
            run_name = f"meta_run_{current_time}"
            
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run_name
                logging.info(f"Started MLflow run: {run_id}")
                
                # Log meta model info
                mlflow.log_param("model_name", self.model_name)
                mlflow.log_param("source_table", table_name)
                mlflow.log_param("base_models", list(self.base_predictors.keys()))
                
                # Get predictions from base models
                base_predictions = {}
                for model_name, predictor in self.base_predictors.items():
                    with mlflow.start_run(run_name=f"base_{model_name}_{current_time}", nested=True):
                        results_df = predictor.run_predictions(table_name)
                        base_predictions[model_name] = results_df
                
                # Prepare meta features
                meta_features = self.prepare_meta_features(base_predictions)
                
                # Make meta model predictions
                meta_model = xgb.XGBRegressor()
                meta_model.load_model(self.meta_model_info['model_path'])
                scaler = pd.read_pickle(self.meta_model_info['scaler_path'])
                
                # Scale features
                X_scaled = scaler.transform(meta_features)
                meta_predictions = meta_model.predict(X_scaled)
                
                # Create results DataFrame
                results_df = pd.DataFrame()
                results_df['Actual_Price'] = base_predictions[list(self.base_predictors.keys())[0]]['Actual_Price']
                results_df['Predicted_Price'] = meta_predictions
                results_df['Error'] = results_df['Actual_Price'] - results_df['Predicted_Price']
                results_df['Price_Change'] = results_df['Actual_Price'].diff()
                results_df['Predicted_Change'] = results_df['Predicted_Price'].diff()
                results_df['Price_Volatility'] = results_df['Price_Change'].rolling(window=20).std()
                
                # Generate summary metrics
                summary = self.base_predictors[list(self.base_predictors.keys())[0]].generate_summary(results_df)
                
                # Log metrics to MLflow
                for metric_name, metric_value in summary.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)
                
                # Store predictions and metrics
                base_predictor = self.base_predictors[list(self.base_predictors.keys())[0]]
                
                # Create a copy of the base predictor with meta model name
                meta_predictor = base_predictor
                meta_predictor.model_predictor.current_model_name = self.model_name
                
                # Store predictions and metrics with meta model name
                meta_predictor.store_predictions(
                    results_df, summary, table_name, run_id
                )
                meta_predictor.store_metrics(
                    summary, run_id, table_name
                )
                
                logging.info(f"Successfully ran meta model predictions on {table_name}")
                return results_df
                
        except Exception as e:
            logging.error(f"Error running meta model predictions: {e}")
            raise

def run_meta_predictions(table_name: str, meta_model_name: str = None, force_new_run: bool = False) -> Dict:
    """Run meta model predictions on a table
    
    Args:
        table_name: Name of the table to run predictions on
        meta_model_name: Name of the meta model to use (if None, will use latest)
        force_new_run: Whether to force a new prediction run even if predictions exist
    
    Returns:
        Dictionary containing results and metrics
    """
    try:
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        models_dir = os.path.join(current_dir, 'models')
        
        # Initialize predictor
        predictor = MetaModelPredictor(db_path, models_dir, meta_model_name)
        
        # Run predictions
        results_df = predictor.run_predictions(table_name)
        
        # Print summary
        print("\nMeta Model Prediction Summary:")
        print(f"Total predictions: {len(results_df)}")
        print(f"Using meta model: {meta_model_name}")
        print(f"Base models: {list(predictor.base_predictors.keys())}")
        print()
        print("Error Metrics:")
        print(f"Mean absolute error: {results_df['Error'].abs().mean():.4f}")
        print(f"Root mean squared error: {np.sqrt((results_df['Error']**2).mean()):.4f}")
        print(f"Mean absolute percentage error: {(abs(results_df['Error']/results_df['Actual_Price'])*100).mean():.2f}%")
        
        # Calculate direction accuracy
        correct_direction = (results_df['Price_Change'] * results_df['Predicted_Change'] > 0).sum()
        total_predictions = len(results_df) - 1  # Subtract 1 because first change is NaN
        direction_accuracy = (correct_direction / total_predictions) * 100
        print(f"Direction Accuracy: {direction_accuracy:.2f}%")
        
        return {
            'results_df': results_df,
            'metrics': {
                'mae': results_df['Error'].abs().mean(),
                'rmse': np.sqrt((results_df['Error']**2).mean()),
                'mape': (abs(results_df['Error']/results_df['Actual_Price'])*100).mean(),
                'direction_accuracy': direction_accuracy
            }
        }
        
    except Exception as e:
        logging.error(f"Error in meta model predictions: {str(e)}")
        logging.exception("Detailed traceback:")
        raise

def main():
    """Main function for testing meta model predictions"""
    try:
        setup_logging()
        logging.info("Starting meta model predictions...")
        
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        models_dir = os.path.join(current_dir, 'models')
        
        # Get command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='Run meta model predictions')
        parser.add_argument('--table', type=str, required=True, help='Table name to run predictions on')
        parser.add_argument('--model', type=str, help='Meta model name (if not provided, will use latest)')
        parser.add_argument('--force', action='store_true', help='Force new prediction run')
        args = parser.parse_args()
        
        # Run predictions
        results = run_meta_predictions(
            table_name=args.table,
            meta_model_name=args.model,
            force_new_run=args.force
        )
        
        logging.info("Meta model predictions completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        logging.exception("Detailed traceback:")
        raise

if __name__ == "__main__":
    main() 