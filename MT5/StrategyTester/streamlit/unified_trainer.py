from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from datetime import datetime
import logging
from model_base import BaseModel

class UnifiedModelTrainer:
    def __init__(self, db_path: str, models_dir: str):
        self.db_path = db_path
        self.models_dir = models_dir
        self.setup_logging()
        
    def train_model(self, 
                   model: BaseModel,
                   table_names: List[str],
                   target_col: str,
                   feature_cols: List[str],
                   force_retrain: bool = False,
                   **kwargs) -> Tuple[str, Dict]:
        """Unified training method that handles both incremental and full training"""
        try:
            # Load data
            df = self.load_data_from_tables(table_names)
            
            # Prepare features
            X, y = self.prepare_features(df, target_col, feature_cols)
            
            # Check if we should do incremental training
            if not force_retrain and model.supports_incremental and self.has_existing_model(model):
                try:
                    # Attempt incremental training
                    metrics = model.partial_fit(X, y, **kwargs)
                    model_path = model.save(self.models_dir)
                    
                    self._update_training_history(model_path, table_names, metrics, 'incremental')
                    return model_path, metrics
                    
                except Exception as e:
                    logging.warning(f"Incremental training failed, falling back to full training: {e}")
                    
            # Full training
            model, metrics = model.train(X, y, **kwargs)
            model_path = model.save(self.models_dir)
            
            self._update_training_history(model_path, table_names, metrics, 'full')
            return model_path, metrics
            
        except Exception as e:
            logging.error(f"Error in unified training: {e}")
            raise 