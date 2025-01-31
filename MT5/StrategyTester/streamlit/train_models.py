import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import json
import pandas as pd
from model_trainer import TimeSeriesModelTrainer
from feature_config import SELECTED_FEATURES
from unified_trainer import UnifiedModelTrainer
from model_factory import ModelFactory

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )



# def get_model_params(model_type: str) -> Dict:
#     """Get model parameters based on model type"""
#     params = {
#         'xgboost': {
#             'max_depth': 8,
#             'learning_rate': 0.05,
#             'n_estimators': 1000,
#             'subsample': 0.8,
#             'colsample_bytree': 0.8,
#             'min_child_weight': 2,
#             'objective': 'reg:squarederror',
#             'random_state': 42
#         },
#         'decision_tree': {
#             'max_depth': 8,
#             'min_samples_split': 5,
#             'min_samples_leaf': 2,
#             'random_state': 42
#         }
#     }
#     return params.get(model_type, {})


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
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
    }
    return params.get(model_type, {})

def generate_model_name(model_type: str, training_type: str, timestamp: Optional[str] = None) -> str:
    """Generate consistent model name
    
    Args:
        model_type: Type of model (e.g., 'xgboost', 'decision_tree')
        training_type: Type of training ('single', 'multi', 'incremental', 'base')
        timestamp: Optional timestamp, will generate if None
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_type}_{training_type}_{timestamp}"

def train_single_table(table_name: str, force_retrain: bool = False):
    """Train a model using a single table"""
    try:
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        models_dir = os.path.join(current_dir, 'models')
        
        # Initialize trainer
        trainer = TimeSeriesModelTrainer(db_path=db_path, models_dir=models_dir)
        
        # Get configurations
        configurations = [
            {'model_type': 'xgboost', 'prediction_horizon': 1},
            {'model_type': 'decision_tree', 'prediction_horizon': 1},
            {'model_type': 'random_forest', 'prediction_horizon': 1}
        ]
        
        results = {}
        for config in configurations:
            model_type = config['model_type']
            logging.info(f"\nTraining {model_type} model for table: {table_name}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = generate_model_name(model_type, 'single', timestamp) if force_retrain else None
            
            try:
                model_path, metrics = trainer.train_and_save_multi_table(
                    table_names=[table_name],
                    target_col="Price",
                    feature_cols=SELECTED_FEATURES,
                    prediction_horizon=config['prediction_horizon'],
                    model_params=get_model_params(model_type),
                    model_name=model_name,
                    model_type=model_type 
                )
                
                # results[(model_type, config['prediction_horizon'])] = {
                #     'model_path': model_path,
                #     'metrics': metrics
                # }
                results[f"{model_type}_h{config['prediction_horizon']}"] = {
                    'model_path': model_path,
                    'metrics': metrics,
                    'model_type': model_type,
                    'prediction_horizon': config['prediction_horizon']
                }
                
                logging.info(f"Training completed: {model_path}")
                logging.info(f"Metrics: {metrics}")
                
            except Exception as e:
                logging.error(f"Error training {model_type} model: {e}")
                continue
                
        return results
        
    except Exception as e:
        logging.error(f"Error in single table training: {str(e)}")
        logging.exception("Detailed traceback:")
        raise

def train_multi_table(table_names: List[str], force_retrain: bool = False):
    """Train a model using multiple tables simultaneously"""
    try:
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        models_dir = os.path.join(current_dir, 'models')
        
        # Initialize trainer
        trainer = TimeSeriesModelTrainer(db_path=db_path, models_dir=models_dir)
        
        # Get configurations
        configurations = [
            # {'model_type': 'xgboost', 'prediction_horizon': 1},
            # {'model_type': 'decision_tree', 'prediction_horizon': 1},
            {'model_type': 'random_forest', 'prediction_horizon': 1}
        ]
        
        results = {}
        for config in configurations:
            model_type = config['model_type']
            logging.info(f"\nProcessing {model_type} model with tables: {table_names}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = generate_model_name(model_type, 'multi', timestamp) if force_retrain else None
            
            try:
                # Train model with multiple tables
                model_path, metrics = trainer.train_and_save_multi_table(
                    table_names=table_names,
                    target_col="Price",
                    feature_cols=SELECTED_FEATURES,
                    prediction_horizon=config['prediction_horizon'],
                    model_params=get_model_params(model_type),
                    model_name=model_name,
                    model_type=model_type 
                )
                
                # results[(model_type, config['prediction_horizon'])] = {
                #     'model_path': model_path,
                #     'metrics': metrics
                # }
                results[f"{model_type}_h{config['prediction_horizon']}"] = {
                    'model_path': model_path,
                    'metrics': metrics,
                    'model_type': model_type,
                    'prediction_horizon': config['prediction_horizon']
                }
                
                logging.info(f"Training completed: {model_path}")
                logging.info(f"Metrics: {metrics}")
                
            except Exception as e:
                logging.error(f"Error training {model_type} model: {e}")
                continue
                
        return results
        
    except Exception as e:
        logging.error(f"Error in multi-table training: {str(e)}")
        logging.exception("Detailed traceback:")
        raise

def train_model_incrementally(base_table: str, new_tables: List[str], force_retrain: bool = False):
    """
    Train a model incrementally using a base table and new tables
    
    Args:
        base_table: Initial table to train on
        new_tables: List of new tables to incrementally train with
        force_retrain: Whether to force a full retrain instead of incremental
    """
    try:
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        models_dir = os.path.join(current_dir, 'models')
        
        # Initialize trainer
        trainer = TimeSeriesModelTrainer(db_path=db_path, models_dir=models_dir)
        
        # First, train on base table
        logging.info(f"Initial training on base table: {base_table}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_model_name = generate_model_name('xgboost', 'base', timestamp)
        # base_model_name =  "model_20250130_181424"

        model_type = 'xgboost'  # Define model type explicitly
        
        model_path, metrics = trainer.train_and_save_multi_table(
            table_names=[base_table],
            target_col="Price",
            feature_cols=SELECTED_FEATURES,
            prediction_horizon=1,
            model_params=get_model_params('xgboost'),
            model_name=base_model_name,
            model_type=model_type 
        )
        
        logging.info(f"Base model trained: {model_path}")
        logging.info(f"Base metrics: {metrics}")
        
        # Store results
        results = {
            'base_training': {
                'model_path': model_path,
                'metrics': metrics
            },
            'incremental_updates': []
        }
        
        # Incrementally train on new tables
        for i, table in enumerate(new_tables, 1):
            logging.info(f"Incremental training {i}/{len(new_tables)} with table: {table}")
            
            try:
                updated_path, updated_metrics = trainer.train_and_save_multi_table(
                    table_names=[table],
                    target_col="Price",
                    feature_cols=SELECTED_FEATURES,
                    prediction_horizon=1,
                    model_params=get_model_params('xgboost'),
                    model_name=base_model_name  # Keep same name for incremental updates
                )
                
                results['incremental_updates'].append({
                    'table': table,
                    'model_path': updated_path,
                    'metrics': updated_metrics
                })
                
                logging.info(f"Incremental update completed: {updated_path}")
                logging.info(f"Updated metrics: {updated_metrics}")
                
            except Exception as e:
                logging.error(f"Error during incremental training on table {table}: {e}")
                if not force_retrain:
                    raise
                    
                logging.info("Attempting full retrain due to force_retrain=True")
                # If force_retrain is True, do a full retrain with all data up to this point
                all_tables = [base_table] + new_tables[:i+1]
                updated_path, updated_metrics = trainer.train_and_save_multi_table(
                    table_names=all_tables,
                    target_col="Price",
                    feature_cols=SELECTED_FEATURES,
                    prediction_horizon=1,
                    model_params=get_model_params('xgboost')
                )
                
                results['incremental_updates'].append({
                    'table': table,
                    'model_path': updated_path,
                    'metrics': updated_metrics,
                    'retrained': True
                })
        
        return results
        
    except Exception as e:
        logging.error(f"Error in incremental training: {str(e)}")
        logging.exception("Detailed traceback:")
        raise

def train_model(table_name: str, model_type: str = 'xgboost', force_retrain: bool = False):
    """High-level training function"""
    try:
        # Setup
        trainer = UnifiedModelTrainer(db_path='path/to/db', models_dir='path/to/models')
        
        # Create model instance
        model = ModelFactory.create(model_type)
        
        # Train
        model_path, metrics = trainer.train_model(
            model=model,
            table_names=[table_name],
            target_col="Price",
            feature_cols=SELECTED_FEATURES,
            force_retrain=force_retrain
        )
        
        return model_path, metrics
        
    except Exception as e:
        logging.error(f"Error in training: {e}")
        raise

if __name__ == "__main__":
    setup_logging()
    
    # Example usage of different training methods
    try:
        
        # # 1. Single table training
        # single_table = "strategy_TRIP_NAS_10019851"
        # single_results = train_single_table(single_table)
        # logging.info("\nSingle Table Training Results:================================================================")
        # for model_key, result in single_results.items():
        #     logging.info(f"\nModel: {model_key}")
        #     logging.info(f"Model Path: {result['model_path']}")
        #     logging.info(f"Metrics: {result['metrics']}")
        
        # 2. Multi-table training
        multiple_tables = [
            "strategy_TRIP_NAS_10019851",
            "strategy_TRIP_NAS_10031622",
            "strategy_TRIP_NAS_10026615"
        ]
        multi_results = train_multi_table(multiple_tables)
        logging.info("\nMulti-Table Training Results:================================================================")
        for model_key, result in multi_results.items():
            logging.info(f"\nModel: {model_key}")
            logging.info(f"Model Path: {result['model_path']}")
            logging.info(f"Metrics: {result['metrics']}")
        
        # # 3. Incremental training
        # base_table = "strategy_TRIP_NAS_10019851"
        # new_tables = [
        #     "strategy_TRIP_NAS_10031622",
        #     "strategy_TRIP_NAS_10026615"
        # ]
        # incremental_results = train_model_incrementally(base_table, new_tables)

        # logging.info("\nIncremental Training Results:================================================================")
        # logging.info("\nBase Training:")
        # logging.info(f"Model Path: {incremental_results['base_training']['model_path']}")
        # logging.info(f"Metrics: {incremental_results['base_training']['metrics']}")
        
        # for i, update in enumerate(incremental_results['incremental_updates'], 1):
        #     logging.info(f"\nIncremental Update {i}:")
        #     logging.info(f"Table: {update['table']}")
        #     logging.info(f"Model Path: {update['model_path']}")
        #     logging.info(f"Metrics: {update['metrics']}")
        #     if update.get('retrained'):
        #         logging.info("Note: Full retrain was performed for this update")
        
    except Exception as e:
        logging.error(f"Critical error in training script: {str(e)}")
        logging.exception("Detailed traceback:")
        raise