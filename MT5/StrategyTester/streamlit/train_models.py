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
import torch

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
        },
        'lstm': {
            'hidden_size': 64,
            'num_layers': 2,
            'sequence_length': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 10,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
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

def train_single_table(table_name: str, force_retrain: bool = False, model_types: List[str] = ['lstm']):
    """Train a model using a single table"""
    try:
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        models_dir = os.path.join(current_dir, 'models')
        
        # Initialize trainer
        trainer = TimeSeriesModelTrainer(db_path=db_path, models_dir=models_dir)
        
        # Get configurations based on model_types parameter
        configurations = [{'model_type': model_type, 'prediction_horizon': 1} for model_type in model_types]
        
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

def train_multi_table(table_names: List[str], force_retrain: bool = False, model_types: List[str] = ['lstm']):
    """Train a model using multiple tables simultaneously"""
    try:
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        models_dir = os.path.join(current_dir, 'models')
        
        # Initialize trainer
        trainer = TimeSeriesModelTrainer(db_path=db_path, models_dir=models_dir)
        
        # Get configurations based on model_types parameter
        configurations = [{'model_type': model_type, 'prediction_horizon': 1} for model_type in model_types]
        
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

def train_model_incrementally(base_table: str, new_tables: List[str], force_retrain: bool = False, model_types: List[str] = ['xgboost']):
    """
    Train a model incrementally using a base table and new tables
    
    Args:
        base_table: Initial table to train on
        new_tables: List of new tables to incrementally train with
        force_retrain: Whether to force a full retrain instead of incremental
        model_types: List of model types to train (default: ['xgboost'])
    """
    try:
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        models_dir = os.path.join(current_dir, 'models')
        
        # Initialize trainer
        trainer = TimeSeriesModelTrainer(db_path=db_path, models_dir=models_dir)
        
        results = {'base_training': {}, 'incremental_updates': []}
        
        for model_type in model_types:
            logging.info(f"\nTraining {model_type} model incrementally")
            
            # First, train on base table
            logging.info(f"Initial training on base table: {base_table}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_model_name = generate_model_name(model_type, 'base', timestamp)
            
            # Train base model
            model_path, metrics = trainer.train_and_save_multi_table(
                table_names=[base_table],
                target_col="Price",
                feature_cols=SELECTED_FEATURES,
                prediction_horizon=1,
                model_params=get_model_params(model_type),
                model_name=base_model_name,
                model_type=model_type
            )
            
            results['base_training'][model_type] = {
                'model_path': model_path,
                'metrics': metrics
            }
            
            # Then incrementally update with new tables
            for new_table in new_tables:
                logging.info(f"Incrementally training with table: {new_table}")
                
                try:
                    # Load the latest model and update it
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    incremental_model_name = generate_model_name(model_type, 'incremental', timestamp)
                    
                    # Use the previous model path for incremental training
                    model_path, metrics = trainer.train_and_save_multi_table(
                        table_names=[new_table],
                        target_col="Price",
                        feature_cols=SELECTED_FEATURES,
                        prediction_horizon=1,
                        model_params=get_model_params(model_type),
                        model_name=incremental_model_name,
                        model_type=model_type,
                        base_model_path=model_path if not force_retrain else None
                    )
                    
                    results['incremental_updates'].append({
                        'model_type': model_type,
                        'table': new_table,
                        'model_path': model_path,
                        'metrics': metrics
                    })
                    
                except Exception as e:
                    logging.error(f"Error in incremental update for table {new_table}: {e}")
                    if force_retrain:
                        logging.info("Attempting full retrain...")
                        # If incremental update fails and force_retrain is True, do a full retrain
                        model_path, metrics = trainer.train_and_save_multi_table(
                            table_names=[base_table] + new_tables[:new_tables.index(new_table) + 1],
                            target_col="Price",
                            feature_cols=SELECTED_FEATURES,
                            prediction_horizon=1,
                            model_params=get_model_params(model_type),
                            model_name=incremental_model_name,
                            model_type=model_type
                        )
                        
                        results['incremental_updates'].append({
                            'model_type': model_type,
                            'table': new_table,
                            'model_path': model_path,
                            'metrics': metrics,
                            'retrained': True
                        })
                    else:
                        raise
        
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

def main():
    """Command line interface for model training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train models with different configurations')
    parser.add_argument('--mode', choices=['single', 'multi', 'incremental'], required=True,
                       help='Training mode: single, multi, or incremental')
    parser.add_argument('--tables', nargs='+', required=True,
                       help='Table names for training. For incremental mode, first table is base table.')
    parser.add_argument('--model-types', nargs='+', 
                       choices=['xgboost', 'decision_tree', 'random_forest', 'lstm'],
                       default=['lstm'],
                       help='Model types to train (default: lstm)')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retrain models')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    try:
        if args.mode == 'single':
            if len(args.tables) != 1:
                raise ValueError("Single mode requires exactly one table")
            results = train_single_table(args.tables[0], args.force_retrain, args.model_types)
            
        elif args.mode == 'multi':
            results = train_multi_table(args.tables, args.force_retrain, args.model_types)
            
        elif args.mode == 'incremental':
            if len(args.tables) < 2:
                raise ValueError("Incremental mode requires at least two tables (base table and new tables)")
            base_table = args.tables[0]
            new_tables = args.tables[1:]
            results = train_model_incrementally(base_table, new_tables, args.force_retrain, args.model_types)
        
        # Print results in a readable format
        print("\nTraining Results:")
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()