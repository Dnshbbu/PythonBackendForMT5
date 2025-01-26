# import os
# from xgboost_trainer import TimeSeriesXGBoostTrainer
# import logging
# from typing import Dict, Optional
# from datetime import datetime


import os
from xgboost_trainer import TimeSeriesXGBoostTrainer
import logging
from typing import Dict, Optional
from datetime import datetime  # Add this import
import json  # Add this for handling JSON data
import joblib  # Add this for model serialization
import pandas as pd  # Add this for data handling
import numpy as np  # Add this for numerical operations

def get_model_params() -> Dict:
    """Define model hyperparameters"""
    return {
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 2,
        'objective': 'reg:squarederror',
        'random_state': 42
    }


def get_feature_params() -> Dict:
    """Define feature generation parameters"""
    return {
        'lag_columns': [
            'Price', 'Equity', 'Balance', 'Profit', 
            'Score', 'ExitScore',
            'Factors_srScore', 'Factors_maScore', 'Factors_rsiScore',
            'Factors_macdScore', 'Factors_stochScore', 'Factors_bbScore',
            'ExitFactors_srScore', 'ExitFactors_maScore', 'ExitFactors_rsiScore'
        ],
        'lag_values': [1, 5, 10, 20, 50],  # Multiple lag periods
        
        'rolling_columns': [
            'Price', 'Score', 'ExitScore',
            'Factors_srScore', 'Factors_maScore', 'Factors_rsiScore',
            'Factors_volumeScore', 'Factors_mfiScore'
        ],
        'rolling_windows': [5, 10, 20, 50, 100]  # Multiple rolling windows
    }


# def train_time_series_model(
#     table_name: str,
#     target_col: str = "Price",
#     prediction_horizon: int = 1,
#     custom_feature_params: Optional[Dict] = None,
#     custom_model_params: Optional[Dict] = None
# ) -> tuple:
#     """
#     Train a time series model with specified parameters and features
    
#     Args:
#         table_name: Name of the database table
#         target_col: Target column to predict
#         prediction_horizon: Number of steps ahead to predict
#         custom_feature_params: Optional custom feature parameters
#         custom_model_params: Optional custom model parameters
#     """
#     try:
#         # Setup paths
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
#         model_dir = os.path.join(current_dir, 'models')
        
#         # Initialize trainer
#         trainer = TimeSeriesXGBoostTrainer(db_path, model_dir)
        
#         # Specify the features you want to use
#         selected_features = [
#             # 'Price',  # Basic features
#             'Factors_maScore','Factors_rsiScore','Factors_macdScore','Factors_stochScore','Factors_bbScore','Factors_atrScore','Factors_sarScore','Factors_ichimokuScore','Factors_adxScore','Factors_volumeScore','Factors_mfiScore','Factors_priceMAScore','Factors_emaScore','Factors_emaCrossScore','Factors_cciScore',
#             # 'Factors_maScore','Factors_rsiScore','Factors_macdScore','Factors_stochScore','Factors_bbScore','Factors_atrScore','Factors_sarScore','Factors_ichimokuScore','Factors_adxScore','Factors_volumeScore','Factors_mfiScore','Factors_priceMAScore','Factors_emaScore','Factors_emaCrossScore','Factors_cciScore',
#             'EntryScore_AVWAP','EntryScore_EMA','EntryScore_SR'
#         ]
        
#         # Define feature parameters for the selected columns
#         feature_params = {
#             'lag_columns': selected_features,  # Create lags for these columns
#             'lag_values': [1, 5, 10],  # Create these lag periods
#             'rolling_columns': ['Price', 'Score', 'ExitScore'],  # Create rolling features for these
#             'rolling_windows': [5, 10, 20]  # Rolling windows to use
#         }
        
#         # Get model parameters
#         model_params = custom_model_params or get_model_params()
        
#         # Train and save model
#         model_path, metrics = trainer.train_and_save(
#             table_name=table_name,
#             target_col=target_col,
#             prediction_horizon=prediction_horizon,
#             feature_params=feature_params,
#             feature_cols=selected_features,  # Pass selected features to trainer
#             model_params=model_params
#         )
        
#         # Log results
#         logging.info(f"\nTraining Results:")
#         logging.info(f"Model saved to: {model_path}")
#         logging.info(f"Mean RMSE: {metrics['mean_rmse']:.4f} (±{metrics['std_rmse']:.4f})")
#         logging.info(f"Mean R2: {metrics['mean_r2']:.4f} (±{metrics['std_r2']:.4f})")
        
#         return model_path, metrics
        
#     except Exception as e:
#         logging.error(f"Error in model training: {e}")
#         raise

def train_time_series_model(
    table_name: str,
    target_col: str = "Price",
    prediction_horizon: int = 1,
    custom_feature_params: Optional[Dict] = None,
    custom_model_params: Optional[Dict] = None
) -> tuple:
    """
    Train a time series model with specified parameters and features
    
    Args:
        table_name: Name of the database table
        target_col: Target column to predict
        prediction_horizon: Number of steps ahead to predict
        custom_feature_params: Optional custom feature parameters
        custom_model_params: Optional custom model parameters
    """
    try:
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        model_dir = os.path.join(current_dir, 'models')
        
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize trainer
        trainer = TimeSeriesXGBoostTrainer(db_path, model_dir)
        
        # Specify the features you want to use
        selected_features = [
            'Factors_maScore','Factors_rsiScore','Factors_macdScore','Factors_stochScore',
            'Factors_bbScore','Factors_atrScore','Factors_sarScore','Factors_ichimokuScore',
            'Factors_adxScore','Factors_volumeScore','Factors_mfiScore','Factors_priceMAScore',
            'Factors_emaScore','Factors_emaCrossScore','Factors_cciScore',
            'EntryScore_AVWAP','EntryScore_EMA','EntryScore_SR'
        ]
        
        # Define feature parameters for the selected columns
        feature_params = {
            'lag_columns': selected_features,
            'lag_values': [1, 5, 10],
            'rolling_columns': ['Price', 'Score', 'ExitScore'],
            'rolling_windows': [5, 10, 20]
        }
        
        # Get model parameters
        model_params = custom_model_params or get_model_params()
        
        # Generate unique model name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"xgboost_timeseries_{table_name}_{timestamp}"
        
        # Train and save model
        model_path, metrics = trainer.train_and_save(
            table_name=table_name,
            target_col=target_col,
            prediction_horizon=prediction_horizon,
            feature_params=feature_params,
            feature_cols=selected_features,
            model_params=model_params,
            model_name=model_name  # Pass the model name
        )
        
        # Log results
        logging.info(f"\nTraining Results:")
        logging.info(f"Model saved to: {model_path}")
        logging.info(f"Mean RMSE: {metrics['mean_rmse']:.4f} (±{metrics['std_rmse']:.4f})")
        logging.info(f"Mean R2: {metrics['mean_r2']:.4f} (±{metrics['std_r2']:.4f})")
        
        return model_path, metrics
        
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        raise


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Train models for different prediction horizons
        horizons = [1, 5, 10]  # Predict 1, 5, and 10 steps ahead
        results = {}
        
        for horizon in horizons:
            logging.info(f"\nTraining model for {horizon}-step ahead prediction")
            model_path, metrics = train_time_series_model(
                table_name="strategy_SYM_10021279",  # Replace with your table name
                target_col="Price",
                prediction_horizon=horizon
            )
            results[horizon] = {
                'model_path': model_path,
                'metrics': metrics
            }
        
        # Print summary
        logging.info("\nTraining Summary:")
        for horizon, result in results.items():
            logging.info(f"\n{horizon}-step ahead prediction:")
            logging.info(f"Model: {os.path.basename(result['model_path'])}")
            logging.info(f"RMSE: {result['metrics']['mean_rmse']:.4f} (±{result['metrics']['std_rmse']:.4f})")
            logging.info(f"R2: {result['metrics']['mean_r2']:.4f} (±{result['metrics']['std_r2']:.4f})")
            
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()