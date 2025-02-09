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
                # Create the experiment if it doesn't exist
                experiment_id = mlflow.create_experiment("model_predictions")
                logging.info(f"Created new MLflow experiment with ID: {experiment_id}")
            else:
                logging.info(f"Using existing MLflow experiment with ID: {experiment.experiment_id}")
            
            mlflow.set_experiment("model_predictions")
            
            # Create run name
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]  # Include milliseconds but truncate to 3 digits
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
                meta_model.load_model(self.meta_model_info['model_path'])  # Load JSON format model
                scaler = pd.read_pickle(self.meta_model_info['scaler_path'])  # Load pickled scaler
                
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

def main():
    """Main function for testing meta model predictions"""
    try:
        logging.basicConfig(level=logging.INFO)
        logging.info("Starting meta model predictions...")
        
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        models_dir = os.path.join(current_dir, 'models')
        
        logging.info(f"Using database: {db_path}")
        logging.info(f"Using models directory: {models_dir}")
        
        # Parameters
        model_name = "meta_xgboost_20250209_112236"  # Our newly trained meta model
        table_name = "strategy_TRIP_NAS_10031622"  # The table you specified
        
        logging.info(f"Meta model name: {model_name}")
        logging.info(f"Target table: {table_name}")
        
        # Initialize predictor
        logging.info("Initializing MetaModelPredictor...")
        predictor = MetaModelPredictor(db_path, models_dir, model_name)
        
        # Run predictions
        logging.info("Running predictions...")
        results_df = predictor.run_predictions(table_name)
        
        # Print summary
        print("\nMeta Model Prediction Summary:")
        print(f"Total predictions: {len(results_df)}")
        print(f"Using meta model: {model_name}")
        print(f"Base models: {list(predictor.base_predictors.keys())}")
        print()
        print("Error Metrics:")
        print(f"Mean absolute error: {results_df['Error'].abs().mean():.4f}")
        print(f"Root mean squared error: {np.sqrt((results_df['Error'] ** 2).mean()):.4f}")
        print(f"Mean absolute percentage error: {(np.abs(results_df['Error'] / results_df['Actual_Price']) * 100).mean():.2f}%")
        
        # Direction accuracy
        correct_direction = (
            (results_df['Price_Change'] > 0) & (results_df['Predicted_Change'] > 0) |
            (results_df['Price_Change'] < 0) & (results_df['Predicted_Change'] < 0)
        )
        print(f"\nDirection Accuracy: {correct_direction.mean():.2%}")
        
        logging.info("Meta model predictions completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main() 