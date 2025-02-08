import os
from datetime import datetime
import logging
from meta_model_trainer import MetaModelTrainer

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    # Set parameters
    db_path = "logs/trading_data.db"
    base_models_dir = "models"
    test_size = 0.2
    
    # Specific run IDs we want to use
    run_ids = ['run_20250207_234606', 'run_20250207_204318']
    
    # Setup logging
    setup_logging()
    
    # Setup directory structure
    os.makedirs(base_models_dir, exist_ok=True)
    meta_models_dir = os.path.join(base_models_dir, 'meta_models')
    os.makedirs(meta_models_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = MetaModelTrainer(db_path, meta_models_dir)
    
    # Get the date range from the available predictions
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MIN(datetime) as start_date, MAX(datetime) as end_date
            FROM historical_predictions
            WHERE run_id IN (?, ?)
        """, run_ids)
        min_date, max_date = cursor.fetchone()
        start_date = datetime.strptime(min_date, '%Y-%m-%d %H:%M:%S')
        end_date = datetime.strptime(max_date, '%Y-%m-%d %H:%M:%S')
        
    logging.info(f"\nTraining meta-model using predictions from {start_date} to {end_date}")
    logging.info(f"Using run_ids: {run_ids}")
    
    # Train meta-model
    metrics = trainer.train_meta_model(
        start_date=start_date,
        end_date=end_date,
        run_ids=run_ids,
        test_size=test_size
    )
    
    # Log results
    logging.info("\nMeta-model training completed")
    logging.info("\nTraining metrics:")
    for metric_name, value in metrics.items():
        logging.info(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main() 