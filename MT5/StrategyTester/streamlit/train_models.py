import os
import logging
from datetime import datetime
from model_training_manager import ModelTrainingManager
from model_manager import ModelManager
from typing import Optional, Dict, Any
import json

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class IncrementalTrainer:
    def __init__(self, db_path: str, models_dir: str):
        self.db_path = db_path
        self.models_dir = models_dir
        self.model_manager = ModelManager()
        self.training_manager = ModelTrainingManager(
            db_path=db_path,
            models_dir=models_dir,
            min_rows_for_training=20
        )
        
        # Track training history
        self.history_file = os.path.join(models_dir, 'training_history.json')
        self.training_history = self.load_training_history()
        
    def load_training_history(self) -> Dict[str, Any]:
        """Load training history from file"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {'models': {}}
        
    def save_training_history(self):
        """Save training history to file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.training_history, f, indent=4)
            
    def update_training_history(self, model_type: str, metrics: Dict[str, Any], 
                              training_type: str = 'full'):
        """Update training history with new metrics"""
        if model_type not in self.training_history['models']:
            self.training_history['models'][model_type] = []
            
        entry = {
            'timestamp': datetime.now().isoformat(),
            'training_type': training_type,
            'metrics': metrics
        }
        self.training_history['models'][model_type].append(entry)
        self.save_training_history()

    def train_models(self, table_name: str, force_retrain: bool = False):
        """Train or incrementally update models"""
        try:
            # Get configurations to try
            configurations = [
                {'model_type': 'xgboost', 'prediction_horizon': 1},
                # {'model_type': 'decision_tree', 'prediction_horizon': 1}
            ]
            
            for config in configurations:
                model_type = config['model_type']
                logging.info(f"\nProcessing {model_type} model")
                
                # Set active model
                self.model_manager.set_active_model(model_type)
                
                # Get model instance
                model = self.training_manager.get_model_instance(model_type)
                
                # Check if model supports incremental learning
                try:
                    if not force_retrain and model.supports_incremental_learning():
                        logging.info(f"{model_type} supports incremental learning")
                        self.handle_incremental_training(model, table_name, config)
                    else:
                        logging.info(f"{model_type} requires full training")
                        self.handle_full_training(model, table_name, config)
                except AttributeError as e:
                    logging.warning(f"{model_type} does not implement incremental learning interface, defaulting to full training")
                    self.handle_full_training(model, table_name, config)
                    
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise


    def handle_incremental_training(self, model, table_name: str, config: Dict):
        """Handle incremental training for supported models"""
        try:
            # Check if model file exists
            model_files = [f for f in os.listdir(self.models_dir) if f.startswith(config['model_type']) and f.endswith('.joblib')]
            
            if not model_files:
                logging.info(f"No existing {config['model_type']} model found. Performing full training.")
                self.handle_full_training(model, table_name, config)
                return

            # Get new data since last training
            new_data = self.training_manager.get_new_data(table_name)
            
            if new_data.empty:
                logging.info("No new data available for training")
                return
                
            if model.needs_retraining(len(new_data)):
                logging.info("Model needs full retraining based on data size")
                self.handle_full_training(model, table_name, config)
                return
                
            # Prepare new data for incremental training
            X_new, y_new = self.training_manager.prepare_features_target(
                new_data,
                config['prediction_horizon']
            )
            
            # Perform incremental training
            metrics = model.partial_fit(X_new, y_new)
            
            # Update training history
            self.update_training_history(
                config['model_type'],
                metrics,
                'incremental'
            )
            
            # Save updated model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = model.save(self.models_dir, timestamp)
            
            logging.info(f"Incremental training completed: {model_path}")
            logging.info(f"Metrics: {metrics}")
            
        except FileNotFoundError as fe:
            logging.error(f"Model or directory not found: {fe}")
            self.handle_full_training(model, table_name, config)
        except PermissionError as pe:
            logging.error(f"Permission error accessing model files: {pe}")
        except Exception as e:
            logging.error(f"Error in incremental training: {e}")
            raise

    def handle_full_training(self, model, table_name: str, config: Dict):
        """Handle full model retraining"""
        try:
            # Trigger full training
            result = self.training_manager.retrain_model(table_name)
            
            # Debug log
            logging.info(f"Retrain model returned: {result}")
            
            if result is None or result == (None, None):
                logging.warning("Training skipped or failed") 
                return
                
            model_path, metrics = result
            if model_path is None or metrics is None:
                logging.warning("No valid model path or metrics returned")
                return
                
            # Update training history
            self.update_training_history(
                config['model_type'],
                metrics,
                'full'
            )
            
            logging.info(f"Full training completed: {model_path}")
            logging.info(f"Training metrics: {metrics}")
            
        except ValueError as ve:
            logging.error(f"Value error in full training: {ve}")
        except TypeError as te:
            logging.error(f"Type error in full training: {te}")
        except Exception as e:
            logging.error(f"Error in full training: {e}")
            raise


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
        
        # Initialize trainer
        trainer = IncrementalTrainer(db_path, models_dir)
        
        # Train models
        # table_name = "strategy_TRIP_NAS_10019851"  # Replace with your table name
        # table_name = "strategy_TRIP_NAS_10031622"  # Replace with your table name
        table_name = "strategy_TRIP_NAS_10026615"  # Replace with your table name
        trainer.train_models(table_name)
        
    except Exception as e:
        logging.error(f"Critical error in training script: {str(e)}")
        raise

if __name__ == "__main__":
    main()