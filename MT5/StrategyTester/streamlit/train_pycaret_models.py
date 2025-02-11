import os
import logging
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import json
import pandas as pd
from pycaret_model_trainer import PyCaretModelTrainer
from db_info import get_table_names, get_numeric_columns

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def generate_model_name(timestamp: Optional[str] = None) -> str:
    """Generate a model name with timestamp"""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"pycaret_model_{timestamp}"

def train_model(
    table_names: List[str],
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    prediction_horizon: int = 1,
    model_name: Optional[str] = None,
    model_params: Optional[Dict] = None
) -> Tuple[str, Dict]:
    """
    Train a PyCaret model on the specified data
    
    Args:
        table_names: List of tables to use for training
        target_col: Target column to predict
        feature_cols: Optional list of feature columns
        prediction_horizon: Steps ahead to predict
        model_name: Optional name for the model
        model_params: Optional model parameters
        
    Returns:
        Tuple of (model directory path, metrics dictionary)
    """
    db_path = "trading_data.db"
    models_dir = "models"
    
    trainer = PyCaretModelTrainer(db_path, models_dir)
    
    if model_name is None:
        model_name = generate_model_name()
        
    logging.info(f"Training PyCaret model with name: {model_name}")
    logging.info(f"Using tables: {', '.join(table_names)}")
    logging.info(f"Target column: {target_col}")
    logging.info(f"Prediction horizon: {prediction_horizon}")
    
    model_dir, metrics = trainer.train_and_save(
        table_names=table_names,
        target_col=target_col,
        prediction_horizon=prediction_horizon,
        feature_cols=feature_cols,
        model_params=model_params,
        model_name=model_name
    )
    
    logging.info("Training completed successfully")
    logging.info(f"Model saved to: {model_dir}")
    logging.info("Performance metrics:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value}")
        
    return model_dir, metrics

def main():
    parser = argparse.ArgumentParser(description='Train PyCaret models for time series prediction')
    
    parser.add_argument(
        '--tables',
        nargs='+',
        help='List of tables to use for training',
        required=True
    )
    
    parser.add_argument(
        '--target',
        help='Target column to predict',
        required=True
    )
    
    parser.add_argument(
        '--features',
        nargs='+',
        help='List of feature columns (optional)',
        default=None
    )
    
    parser.add_argument(
        '--horizon',
        type=int,
        help='Prediction horizon (steps ahead)',
        default=1
    )
    
    parser.add_argument(
        '--model-name',
        help='Name for the trained model (optional)',
        default=None
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        train_model(
            table_names=args.tables,
            target_col=args.target,
            feature_cols=args.features,
            prediction_horizon=args.horizon,
            model_name=args.model_name
        )
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 