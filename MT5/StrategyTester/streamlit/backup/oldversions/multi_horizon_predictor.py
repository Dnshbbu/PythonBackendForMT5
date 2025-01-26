import os
import logging
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import json

class MultiHorizonPredictor:
    def __init__(self, db_path: str, models_dir: str):
        """
        Initialize predictor for multiple prediction horizons
        
        Args:
            db_path: Path to the SQLite database
            models_dir: Directory containing trained models
        """
        self.db_path = db_path
        self.models_dir = models_dir
        self.models = {}  # Dictionary to store models for different horizons
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def load_all_models(self) -> Dict[int, Dict]:
        """
        Load all available models for different prediction horizons
        
        Returns:
            Dictionary mapping horizon to model information
        """
        try:
            # Find all model files
            model_files = []
            for f in os.listdir(self.models_dir):
                if f.endswith('.joblib') and 'scaler' not in f:
                    full_path = os.path.join(self.models_dir, f)
                    model_files.append((full_path, os.path.getctime(full_path)))

            if not model_files:
                raise FileNotFoundError("No models found in models directory")

            # Group models by horizon
            horizon_models = {}
            for model_path, ctime in model_files:
                try:
                    # Extract horizon from filename or metadata
                    base_name = os.path.splitext(os.path.basename(model_path))[0]
                    
                    # Load model metadata to get horizon
                    metadata_path = os.path.join(self.models_dir, f"{base_name}_metrics.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            horizon = metadata.get('prediction_horizon', 1)
                    else:
                        # Default to extracting from filename
                        if 'horizon' in base_name:
                            horizon = int(base_name.split('horizon_')[1].split('_')[0])
                        else:
                            horizon = 1  # Default horizon

                    # Load model and associated files
                    model = joblib.load(model_path)
                    
                    # Load scaler if available
                    scaler_path = os.path.join(self.models_dir, f"{base_name}_scaler.joblib")
                    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
                    
                    # Load feature names
                    feature_names_path = os.path.join(self.models_dir, f"{base_name}_feature_names.json")
                    if os.path.exists(feature_names_path):
                        with open(feature_names_path, 'r') as f:
                            feature_data = json.load(f)
                            feature_columns = feature_data['feature_names']
                    else:
                        feature_columns = model.feature_names_ if hasattr(model, 'feature_names_') else None

                    # Store model info
                    horizon_models[horizon] = {
                        'model': model,
                        'scaler': scaler,
                        'feature_columns': feature_columns,
                        'metadata': metadata if 'metadata' in locals() else {},
                        'creation_time': datetime.fromtimestamp(ctime),
                        'base_name': base_name  # Store base_name for future reference
                    }
                    
                    logging.info(f"Loaded model for {horizon}-step prediction")
                    
                except Exception as e:
                    logging.warning(f"Error loading model {model_path}: {e}")
                    continue

            if not horizon_models:
                raise ValueError("No valid models could be loaded")

            self.models = horizon_models
            return horizon_models

        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise

    def prepare_features(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Prepare features for a specific prediction horizon
        """
        try:
            model_info = self.models[horizon]
            feature_columns = model_info['feature_columns']
            scaler = model_info['scaler']

            # Ensure all required features are present
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                logging.warning(f"Missing features for horizon {horizon}: {missing_features}")
                for col in missing_features:
                    df[col] = 0

            # Select and order features
            X = df[feature_columns].copy()

            # Scale features if scaler exists
            if scaler:
                try:
                    X_scaled = pd.DataFrame(
                        scaler.transform(X),
                        columns=feature_columns,
                        index=X.index
                    )
                except Exception as e:
                    logging.error(f"Error scaling features for horizon {horizon}: {e}")
                    raise
            else:
                X_scaled = X

            return X_scaled

        except Exception as e:
            logging.error(f"Error preparing features for horizon {horizon}: {e}")
            raise

    def make_all_predictions(self, table_name: str, n_rows: int = 100, 
                           confidence_threshold: float = 0.8) -> Dict[int, Dict]:
        """
        Make predictions using all available models
        
        Returns:
            Dictionary mapping horizon to prediction results
        """
        try:
            # Load data
            query = f"""
            SELECT *
            FROM {table_name}
            ORDER BY id DESC
            LIMIT {n_rows}
            """
            
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                raise ValueError("No data available for prediction")

            # Make predictions for each horizon
            all_predictions = {}
            for horizon, model_info in self.models.items():
                try:
                    # Prepare features
                    X = self.prepare_features(df, horizon)
                    
                    # Make prediction
                    model = model_info['model']
                    prediction = model.predict(X.iloc[-1:])
                    
                    # Calculate confidence
                    predictions_sequence = model.predict(X)
                    confidence = 1.0 - np.std(predictions_sequence) / np.mean(np.abs(prediction))
                    
                    # Get feature importance
                    feature_importance = dict(zip(X.columns, model.feature_importances_))
                    top_features = dict(sorted(feature_importance.items(), 
                                             key=lambda x: abs(x[1]), 
                                             reverse=True)[:5])
                    
                    # Store results
                    all_predictions[horizon] = {
                        'timestamp': datetime.now().isoformat(),
                        'prediction': float(prediction[0]),
                        'confidence': float(confidence),
                        'is_confident': confidence >= confidence_threshold,
                        'top_features': top_features,
                        'metadata': {
                            'features_used': len(model_info['feature_columns']),
                            'data_points': len(df),
                            'model_creation_time': model_info['creation_time'].isoformat(),
                            'model_name': model_info['base_name']
                        }
                    }
                    
                except Exception as e:
                    logging.error(f"Error making prediction for horizon {horizon}: {e}")
                    all_predictions[horizon] = {'error': str(e)}

            return all_predictions

        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            raise

    def get_consolidated_explanation(self, predictions: Dict[int, Dict]) -> str:
        """
        Generate a consolidated explanation for all predictions
        """
        try:
            explanations = ["Multi-Horizon Prediction Results:"]
            
            for horizon, result in sorted(predictions.items()):
                if 'error' in result:
                    explanations.append(f"\n{horizon}-step ahead prediction: ERROR - {result['error']}")
                    continue
                    
                explanations.append(f"\n{horizon}-step ahead prediction:")
                explanations.append(f"Prediction: {result['prediction']:.4f}")
                explanations.append(f"Confidence: {result['confidence']*100:.1f}%")
                
                if not result['is_confident']:
                    explanations.append("Warning: Below confidence threshold")
                
                explanations.append("\nTop influencing features:")
                for feature, importance in result['top_features'].items():
                    explanations.append(f"- {feature}: {importance:.4f}")

            return "\n".join(explanations)

        except Exception as e:
            logging.error(f"Error generating explanation: {e}")
            raise

    def get_model_metadata(self) -> Dict[int, Dict]:
        """
        Get metadata for all loaded models
        """
        return {
            horizon: {
                'creation_time': info['creation_time'].isoformat(),
                'features_count': len(info['feature_columns']),
                'has_scaler': info['scaler'] is not None,
                'model_name': info['base_name']
            }
            for horizon, info in self.models.items()
        }