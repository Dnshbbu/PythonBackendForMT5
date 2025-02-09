import os
from datetime import datetime
import logging
import argparse
from meta_model_trainer import MetaModelTrainer
from mlflow_utils import MLflowManager

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_date_range_for_runs(db_path: str, run_ids: list) -> tuple:
    """Get the date range from available predictions for given run IDs"""
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MIN(datetime) as start_date, MAX(datetime) as end_date
            FROM historical_predictions
            WHERE run_id IN ({})
        """.format(','.join(['?' for _ in run_ids])), run_ids)
        min_date, max_date = cursor.fetchone()
        start_date = datetime.strptime(min_date, '%Y-%m-%d %H:%M:%S')
        end_date = datetime.strptime(max_date, '%Y-%m-%d %H:%M:%S')
    return start_date, end_date

def train_meta_model(run_ids: list, test_size: float = 0.2, force_retrain: bool = False) -> dict:
    """Train a meta model using specified run IDs and parameters"""
    try:
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        base_models_dir = os.path.join(current_dir, 'models')
        meta_models_dir = os.path.join(base_models_dir, 'meta_models')
        
        # Create necessary directories
        os.makedirs(base_models_dir, exist_ok=True)
        os.makedirs(meta_models_dir, exist_ok=True)
        
        # Initialize trainer
        trainer = MetaModelTrainer(db_path, meta_models_dir)
        
        # Get date range for the runs
        start_date, end_date = get_date_range_for_runs(db_path, run_ids)
        logging.info(f"\nTraining meta-model using predictions from {start_date} to {end_date}")
        logging.info(f"Using run_ids: {run_ids}")
        
        # Generate model name with timestamp
        model_name = f"meta_xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logging.info(f"Creating meta-model: {model_name}")
        
        # Train meta-model
        metrics = trainer.train_meta_model(
            start_date=start_date,
            end_date=end_date,
            run_ids=run_ids,
            test_size=test_size,
            model_name=model_name
        )
        
        # Get meta model information
        meta_model_info = trainer.model_repository.get_meta_model_info(model_name)
        meta_model_info['metrics'] = metrics
        meta_model_info['model_name'] = model_name
        
        return meta_model_info
        
    except Exception as e:
        logging.error(f"Error in meta-model training: {str(e)}")
        logging.exception("Detailed traceback:")
        raise

def main():
    """Command line interface for meta-model training"""
    parser = argparse.ArgumentParser(description='Train meta-model with specified parameters')
    parser.add_argument('--run-ids', nargs='+', required=True,
                       help='List of run IDs to use for meta-model training')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test size for model validation (default: 0.2)')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retrain even if model exists')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    try:
        # Train meta-model
        meta_model_info = train_meta_model(
            run_ids=args.run_ids,
            test_size=args.test_size,
            force_retrain=args.force_retrain
        )
        
        # Log results
        logging.info("\nMeta-model training completed")
        logging.info("\nTraining metrics:")
        for metric_name, value in meta_model_info['metrics'].items():
            logging.info(f"{metric_name}: {value:.4f}")
            
        logging.info("\nMeta Model Information:")
        logging.info(f"Base Model Names: {meta_model_info['base_model_names']}")
        logging.info(f"Base Model Run IDs: {meta_model_info['base_model_run_ids']}")
        logging.info(f"Model Type: {meta_model_info['model_type']}")
        logging.info(f"Training Type: {meta_model_info['training_type']}")
        logging.info(f"Number of Features: {len(meta_model_info['features'])}")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 