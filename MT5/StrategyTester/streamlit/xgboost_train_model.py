import os
from model_trainer import TimeSeriesModelTrainer
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime

# Define your feature sets
TECHNICAL_FEATURES = [
    'Factors_maScore', 'Factors_rsiScore', 'Factors_macdScore', 'Factors_stochScore',
    'Factors_bbScore', 'Factors_atrScore', 'Factors_sarScore', 'Factors_ichimokuScore',
    'Factors_adxScore', 'Factors_volumeScore', 'Factors_mfiScore', 'Factors_priceMAScore',
    'Factors_emaScore', 'Factors_emaCrossScore', 'Factors_cciScore'
]

ENTRY_FEATURES = [
    'EntryScore_AVWAP', 'EntryScore_EMA', 'EntryScore_SR'
]

# Combine all features
SELECTED_FEATURES = TECHNICAL_FEATURES + ENTRY_FEATURES

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
        },
        'decision_tree': {
            'max_depth': 8,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
    }
    return params.get(model_type, {})

def train_time_series_model(
    table_name: str,
    model_type: str = "xgboost",
    target_col: str = "Price",
    prediction_horizon: int = 1,
    custom_feature_params: Optional[Dict] = None,
    custom_model_params: Optional[Dict] = None
) -> Tuple[str, Dict]:
    """
    Train a time series model with specified parameters and features
    """
    try:
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        model_dir = os.path.join(current_dir, 'models')
        
        # Initialize trainer
        trainer = TimeSeriesModelTrainer(db_path, model_dir)
        
        # Get model parameters
        model_params = custom_model_params or get_model_params(model_type)
        
        # Train and save model
        model_path, metrics = trainer.train_and_save(
            table_name=table_name,
            model_type=model_type,
            target_col=target_col,
            feature_cols=SELECTED_FEATURES,
            prediction_horizon=prediction_horizon,
            model_params=model_params
        )
        
        # Log results
        logging.info(f"\nTraining Results:")
        logging.info(f"Model Type: {model_type}")
        logging.info(f"Model saved to: {model_path}")
        logging.info(f"Metrics: {metrics}")
        
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
        # Train models with different configurations
        configurations = [
            {'model_type': 'xgboost', 'prediction_horizon': 1},
            {'model_type': 'xgboost', 'prediction_horizon': 5},
            {'model_type': 'decision_tree', 'prediction_horizon': 1},
            {'model_type': 'decision_tree', 'prediction_horizon': 5}
        ]
        
        results = {}
        for config in configurations:
            model_type = config['model_type']
            horizon = config['prediction_horizon']
            
            logging.info(f"\nTraining {model_type} model for {horizon}-step ahead prediction")
            model_path, metrics = train_time_series_model(
                table_name="strategy_SYM_10021279",  # Replace with your table name
                model_type=model_type,
                target_col="Price",
                prediction_horizon=horizon
            )
            
            results[(model_type, horizon)] = {
                'model_path': model_path,
                'metrics': metrics
            }
        
        # Print summary
        logging.info("\nTraining Summary:")
        for (model_type, horizon), result in results.items():
            logging.info(f"\n{model_type.upper()} - {horizon}-step ahead prediction:")
            logging.info(f"Model: {os.path.basename(result['model_path'])}")
            logging.info(f"Training Score: {result['metrics']['training_score']:.4f}")
            logging.info("Top Features by Importance:")
            for feature, importance in sorted(
                result['metrics']['feature_importance'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]:
                logging.info(f"  {feature}: {importance:.4f}")
            
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()