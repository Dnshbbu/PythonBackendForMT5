import os
import logging
from datetime import datetime
from model_training_manager import ModelTrainingManager
from model_manager import ModelManager

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    """Main function for model training"""
    setup_logging()
    
    try:
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        models_dir = os.path.join(current_dir, 'models')
        
        # Ensure directories exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize managers
        model_manager = ModelManager()
        training_manager = ModelTrainingManager(
            db_path=db_path,
            models_dir=models_dir,
            min_rows_for_training=20
        )
        
        # Get configurations to try
        configurations = [
            {'model_type': 'xgboost', 'prediction_horizon': 1},
            {'model_type': 'decision_tree', 'prediction_horizon': 1}
        ]
        
        # Train each configuration
        for config in configurations:
            # Set active model
            model_manager.set_active_model(config['model_type'])
            
            # Update any specific parameters if needed
            # model_manager.update_model_params(config['model_type'], {...})
            
            logging.info(f"\nStarting training for {config['model_type']}")
            
            # Trigger training
            # table_name = "strategy_SYM_10021279"  # Replace with your table name
            table_name = "strategy_TRIP_NAS_10026615"  # Replace with your table name
            training_manager.retrain_model(table_name)
            
            # Get training status
            status = training_manager.get_latest_training_status()
            
            if status.get('status') == 'completed':
                logging.info(f"Training completed successfully")
                logging.info(f"Model saved to: {status.get('model_path')}")
                if status.get('metrics'):
                    logging.info(f"Training metrics: {status.get('metrics')}")
            else:
                logging.error(f"Training failed: {status.get('error_message')}")
    
    except Exception as e:
        logging.error(f"Error in training script: {str(e)}")
        raise

if __name__ == "__main__":
    main()