import os
import logging
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import json
from pycaret.regression import *
from model_repository import ModelRepository

class PyCaretModelPredictor:
    def __init__(self, db_path: str, models_dir: str):
        """
        Initialize the PyCaret ModelPredictor
        
        Args:
            db_path: Path to the SQLite database
            models_dir: Directory containing trained models
        """
        self.db_path = db_path
        self.models_dir = models_dir
        self.setup_logging()
        
        # Initialize model components
        self.model = None
        self.feature_columns = None
        self.current_model_name = None
        
        # Initialize model repository
        self.model_repository = ModelRepository(db_path)
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def load_model_by_name(self, model_name: str) -> None:
        """
        Load a specific PyCaret model by name
        
        Args:
            model_name: Name of the model to load
        """
        model_dir = os.path.join(self.models_dir, model_name)
        model_path = os.path.join(model_dir, "model.pkl")
        metadata_path = os.path.join(model_dir, "metadata.json")
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model not found at {model_path}")
            
        # Load model using PyCaret
        self.model = load_model(model_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        self.feature_columns = metadata['feature_columns']
        self.current_model_name = model_name
        
        logging.info(f"Loaded model {model_name} with {len(self.feature_columns)} features")
        
    def load_latest_model(self) -> None:
        """Load the most recently trained PyCaret model"""
        # Get all model directories
        model_dirs = [d for d in os.listdir(self.models_dir) 
                     if os.path.isdir(os.path.join(self.models_dir, d))
                     and d.startswith("pycaret_model_")]
        
        if not model_dirs:
            raise ValueError("No PyCaret models found")
            
        # Sort by timestamp in name
        latest_model = sorted(model_dirs)[-1]
        self.load_model_by_name(latest_model)
        
    def get_latest_data(self, table_name: str, n_rows: int = 100) -> pd.DataFrame:
        """
        Get the most recent rows from a table
        
        Args:
            table_name: Name of the table to query
            n_rows: Number of rows to retrieve
            
        Returns:
            DataFrame with the most recent data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"""
                SELECT * FROM {table_name}
                ORDER BY Date DESC, Time DESC
                LIMIT {n_rows}
            """
            
            df = pd.read_sql_query(query, conn)
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df = df.set_index('DateTime').sort_index()
            
            return df
            
        finally:
            if conn:
                conn.close()
                
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with prepared features
        """
        if self.feature_columns is None:
            raise ValueError("No model loaded. Call load_model_by_name() first")
            
        # Select only the required feature columns
        X = df[self.feature_columns].copy()
        
        return X
        
    def make_predictions(self, table_name: str, n_rows: int = 100) -> Dict[str, Union[float, str, dict]]:
        """
        Make predictions using the loaded model
        
        Args:
            table_name: Name of the table to get data from
            n_rows: Number of recent rows to use
            
        Returns:
            Dictionary containing prediction results and metadata
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model_by_name() first")
            
        # Get latest data
        df = self.get_latest_data(table_name, n_rows)
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Make prediction using PyCaret
        predictions = predict_model(self.model, data=X)
        latest_prediction = float(predictions.iloc[-1]['prediction_label'])
        
        # Get feature importance if available
        try:
            importance = get_feature_importance(self.model)
            importance_dict = dict(zip(importance['Feature'], importance['Importance']))
        except:
            importance_dict = {}
            
        result = {
            'prediction': latest_prediction,
            'timestamp': datetime.now().isoformat(),
            'model_name': self.current_model_name,
            'feature_importance': importance_dict
        }
        
        return result
        
    def get_prediction_explanation(self, prediction_result: Dict) -> str:
        """
        Generate a human-readable explanation of the prediction
        
        Args:
            prediction_result: Dictionary containing prediction results
            
        Returns:
            String explanation of the prediction
        """
        prediction = prediction_result['prediction']
        importance = prediction_result['feature_importance']
        
        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_features[:3]
        
        explanation = f"The model predicts a value of {prediction:.4f}. "
        
        if top_features:
            explanation += "The most influential features were: "
            feature_explanations = [
                f"{feature} (importance: {importance:.4f})"
                for feature, importance in top_features
            ]
            explanation += ", ".join(feature_explanations)
            
        return explanation 