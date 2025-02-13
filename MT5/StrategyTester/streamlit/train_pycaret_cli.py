import argparse
import logging
import os
import json
from typing import List, Optional
from pycaret_model_trainer import PyCaretModelTrainer
from feature_config import get_feature_groups, get_all_features
from db_info import get_numeric_columns

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pycaret_training.log'),
            logging.StreamHandler()
        ]
    )

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train PyCaret models from command line')
    
    # Required arguments
    parser.add_argument('--table_names', nargs='+', required=True,
                      help='Names of tables to train on')
    parser.add_argument('--target_col', required=True,
                      help='Target column for prediction')
    
    # Optional arguments
    parser.add_argument('--feature_groups', nargs='+', default=['all'],
                      help='Feature groups to use (e.g., price technical volume)')
    parser.add_argument('--prediction_horizon', type=int, default=1,
                      help='Number of steps ahead to predict')
    parser.add_argument('--n_lags', type=int, default=3,
                      help='Number of lagged features to use')
    parser.add_argument('--model_type', default='automl',
                      help='Model type (e.g., automl, xgboost, lightgbm)')
    parser.add_argument('--selected_models', nargs='+',
                      help='Specific models to train in automl mode')
    parser.add_argument('--model_name', 
                      help='Custom name for the trained model')
    parser.add_argument('--use_price_features', action='store_true',
                      help='Whether to use price-based features')
    
    return parser.parse_args()

def get_selected_features(db_path: str, table_name: str, feature_groups: List[str]) -> List[str]:
    """Get features based on selected feature groups"""
    numeric_cols = get_numeric_columns(db_path, table_name)
    
    if 'all' in feature_groups:
        return get_all_features()
    
    selected_features = []
    feature_groups_dict = get_feature_groups()
    
    for group in feature_groups:
        if group in feature_groups_dict:
            group_features = feature_groups_dict[group]
            # Only include features that exist in the table
            selected_features.extend([f for f in group_features if f in numeric_cols])
    
    return list(set(selected_features))

def main():
    """Main function to run PyCaret training from command line"""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logging.info("Starting PyCaret model training")
    
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
    models_dir = os.path.join(current_dir, 'models')
    
    # Initialize trainer
    trainer = PyCaretModelTrainer(db_path, models_dir)
    
    # Get features
    feature_cols = get_selected_features(db_path, args.table_names[0], args.feature_groups)
    
    # Prepare model parameters
    model_params = {
        'model_type': args.model_type
    }
    
    if args.model_type == 'automl' and args.selected_models:
        model_params['selected_models'] = args.selected_models
    
    try:
        # Train model
        model_path, metrics = trainer.train_and_save(
            table_names=args.table_names,
            target_col=args.target_col,
            prediction_horizon=args.prediction_horizon,
            feature_cols=feature_cols,
            model_params=model_params,
            model_name=args.model_name,
            n_lags=args.n_lags,
            use_price_features=args.use_price_features
        )
        
        # Log results
        logging.info(f"Model trained successfully and saved to: {model_path}")
        logging.info("Training metrics:")
        logging.info(json.dumps(metrics, indent=2))
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    main() 