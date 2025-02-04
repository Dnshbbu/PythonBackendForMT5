import json
import os
from typing import Dict, Optional
import logging

class ModelManager:
    def __init__(self, config_path: str = 'model_config.json'):
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def load_config(self) -> Dict:
        """Load model configuration from JSON file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            return self.get_default_config()
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "active_model": "xgboost",
            "models": {
                "xgboost": {
                    "params": {
                        "max_depth": 8,
                        "learning_rate": 0.05,
                        "n_estimators": 1000
                    }
                },
                "decision_tree": {
                    "params": {
                        "max_depth": 8,
                        "min_samples_split": 5,
                        "min_samples_leaf": 2
                    }
                }
            },
            "training": {
                "prediction_horizons": [1, 5],
                "min_rows_for_training": 20
            }
        }
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving config: {e}")
    
    def get_active_model(self) -> str:
        """Get currently active model type"""
        return self.config.get("active_model", "xgboost")
    
    def set_active_model(self, model_type: str):
        """Set active model type"""
        if model_type not in self.config["models"]:
            raise ValueError(f"Unknown model type: {model_type}")
        self.config["active_model"] = model_type
        self.save_config()
    
    def get_model_params(self, model_type: Optional[str] = None) -> Dict:
        """Get parameters for specified model type"""
        model_type = model_type or self.get_active_model()
        return self.config["models"][model_type]["params"].copy()
    
    def update_model_params(self, model_type: str, params: Dict):
        """Update parameters for specified model type"""
        if model_type not in self.config["models"]:
            raise ValueError(f"Unknown model type: {model_type}")
        self.config["models"][model_type]["params"].update(params)
        self.save_config()
    
    def get_training_config(self) -> Dict:
        """Get training configuration"""
        return self.config["training"].copy()
    
    def update_training_config(self, config: Dict):
        """Update training configuration"""
        self.config["training"].update(config)
        self.save_config()