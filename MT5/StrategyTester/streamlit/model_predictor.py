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
from model_implementations import ModelFactory, LSTMModel, TimeSeriesDataset
from model_repository import ModelRepository
import torch

class ModelPredictor:
    def __init__(self, db_path: str, models_dir: str):
        """
        Initialize the ModelPredictor
        
        Args:
            db_path: Path to the SQLite database
            models_dir: Directory containing trained models and scalers
        """
        self.db_path = db_path
        self.models_dir = models_dir
        self.setup_logging()

        # Add model name tracking
        self.current_model_name = None
        
        # Initialize model repository
        self.model_repository = ModelRepository(db_path)
        
        # Initialize model components
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        # Load the latest model by default
        self.load_latest_model()

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def load_model_by_name(self, model_name: str) -> None:
        """Load a specific model by name"""
        try:
            logging.info(f"Loading model: {model_name}")
            
            # Get model info from repository
            cursor = sqlite3.connect(self.db_path).cursor()
            cursor.execute("""
                SELECT features, model_path, model_type, scaler_path 
                FROM model_repository 
                WHERE model_name = ?
            """, (model_name,))
            result = cursor.fetchone()
            
            if not result:
                raise ValueError(f"Model {model_name} not found in repository")
                
            features_json, model_path, model_type, scaler_path = result
            
            # Load features
            self.feature_columns = json.loads(features_json)
            logging.info(f"Loaded {len(self.feature_columns)} features from repository")
            
            # Load model based on type
            if model_type == 'lstm':
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                # Load LSTM model
                checkpoint = torch.load(model_path)
                model_params = checkpoint['model_params']
                
                self.model = LSTMModel(
                    input_size=len(self.feature_columns),
                    hidden_size=model_params['hidden_size'],
                    num_layers=model_params['num_layers']
                ).to(model_params['device'])
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()  # Set to evaluation mode
                
                # Store model parameters for prediction
                self.model_params = model_params
            else:
                # Load traditional models (XGBoost, Random Forest, etc.)
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                self.model = joblib.load(model_path)
            
            self.current_model_name = model_name
            self.model_type = model_type
            logging.info(f"Successfully loaded model: {model_name}")
            
            # Load scaler if available
            if scaler_path and os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logging.info("Successfully loaded scaler")
            else:
                self.scaler = None
                logging.warning("No scaler found for this model")
                
        except Exception as e:
            logging.error(f"Error loading model {model_name}: {e}")
            self.model = None
            self.current_model_name = None
            self.feature_columns = None
            self.scaler = None
            raise

    def load_latest_model(self) -> None:
        """Load the most recent model from the repository"""
        try:
            # Query the latest model from repository
            cursor = sqlite3.connect(self.db_path).cursor()
            cursor.execute("""
                SELECT model_name 
                FROM model_repository 
                WHERE is_active = 1 
                ORDER BY created_at DESC 
                LIMIT 1
            """)
            result = cursor.fetchone()
            
            if not result:
                raise ValueError("No active models found in repository")
                
            latest_model_name = result[0]
            self.load_model_by_name(latest_model_name)
            
        except Exception as e:
            logging.error(f"Error loading latest model: {e}")
            raise

    def get_latest_data(self, table_name: str, n_rows: int = 100) -> pd.DataFrame:
        """
        Fetch the latest n rows from the database
        
        Args:
            table_name: Name of the database table
            n_rows: Number of latest rows to fetch
            
        Returns:
            DataFrame containing the latest rows
        """
        try:
            query = f"""
            SELECT *
            FROM {table_name}
            ORDER BY id DESC
            LIMIT {n_rows}
            """
            
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Convert date and time to datetime
            if 'Date' in df.columns and 'Time' in df.columns:
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                df = df.set_index('DateTime')
            
            df = df.sort_index()
            return df
            
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            raise
            


    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction, ensuring correct feature order
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with prepared features
        """
        try:
            # Log the available columns for debugging
            logging.info(f"Available columns in data: {df.columns.tolist()}")
            logging.info(f"Required feature columns: {self.feature_columns}")
            
            # Ensure all required features are present
            missing_features = [col for col in self.feature_columns if col not in df.columns]
            if missing_features:
                logging.warning(f"Missing features: {missing_features}")
                for col in missing_features:
                    df[col] = 0
            
            # Select and order features according to training order
            X = df[self.feature_columns].copy()
            
            # Log shape before scaling
            logging.info(f"Feature shape before scaling: {X.shape}")
            
            # Scale features if scaler exists
            if self.scaler:
                try:
                    X_scaled = pd.DataFrame(
                        self.scaler.transform(X),
                        columns=self.feature_columns,
                        index=X.index
                    )
                    logging.info("Features scaled successfully")
                except Exception as e:
                    logging.error(f"Error during scaling: {e}")
                    raise
            else:
                X_scaled = X
                
            return X_scaled
            
        except Exception as e:
            logging.error(f"Error preparing features: {e}")
            raise


    def make_predictions(self, table_name: str, n_rows: int = 100, 
                        confidence_threshold: float = 0.8) -> Dict[str, Union[float, str, dict]]:
        """
        Make predictions using the latest data
        
        Args:
            table_name: Name of the database table
            n_rows: Number of latest rows to use
            confidence_threshold: Threshold for confidence level
            
        Returns:
            Dictionary containing prediction results and metadata
        """
        try:
            # Check if model is loaded
            if self.model is None:
                raise ValueError("No model loaded")
                
            # Get latest data
            df = self.get_latest_data(table_name, n_rows)
            if df.empty:
                raise ValueError("No data available for prediction")
                
            # Prepare features
            X = self.prepare_features(df)
            
            # Make prediction based on model type
            if self.model_type == 'lstm':
                # Create sequence dataset
                sequence_length = self.model_params['sequence_length']
                dataset = TimeSeriesDataset(
                    X,
                    pd.Series(np.zeros(len(X))),  # Dummy target
                    sequence_length
                )
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,  # Predict one at a time
                    shuffle=False
                )
                
                # Get last sequence
                last_sequence = next(iter(dataloader))[0]
                
                # Make prediction
                with torch.no_grad():
                    prediction = self.model(last_sequence.to(self.model_params['device']))
                    prediction = prediction.cpu().numpy()
            else:
                # Traditional models
                prediction = self.model.predict(X.iloc[-1:])
            
            # Get feature importance
            if self.model_type == 'lstm':
                # LSTM uses equal feature importance
                importance = 1.0 / len(X.columns)
                feature_importance = dict(zip(X.columns, [importance] * len(X.columns)))
            else:
                feature_importance = dict(zip(X.columns, self.model.feature_importances_))
            
            top_features = dict(sorted(feature_importance.items(), 
                                    key=lambda x: abs(x[1]), 
                                    reverse=True)[:5])
            
            # Calculate prediction confidence
            if self.model_type == 'lstm':
                # For LSTM, use rolling standard deviation of predictions
                confidence = 1.0 - np.std(prediction) / np.mean(np.abs(prediction))
            else:
                confidence = 1.0 - np.std(self.model.predict(X)) / np.mean(np.abs(prediction))
            
            # Prepare result
            result = {
                'timestamp': datetime.now().isoformat(),
                'prediction': float(prediction[0]),
                'confidence': float(confidence),
                'is_confident': confidence >= confidence_threshold,
                'top_features': top_features,
                'model_name': self.current_model_name,
                'model_type': self.model_type,
                'metadata': {
                    'features_used': len(self.feature_columns),
                    'data_points': len(df)
                }
            }
            
            logging.info(f"Made prediction using {self.model_type} model {self.current_model_name}: "
                        f"{result['prediction']:.4f} (confidence: {result['confidence']:.4f})")
            
            return result
            
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            raise

    def get_prediction_explanation(self, prediction_result: Dict) -> str:
        """
        Generate a human-readable explanation of the prediction
        
        Args:
            prediction_result: Dictionary containing prediction results
            
        Returns:
            String containing the explanation
        """
        try:
            explanation = [
                f"Prediction: {prediction_result['prediction']:.4f}",
                f"Confidence: {prediction_result['confidence']*100:.1f}%",
                "\nTop influencing features:"
            ]
            
            for feature, importance in prediction_result['top_features'].items():
                explanation.append(f"- {feature}: {importance:.4f}")
                
            if prediction_result['is_confident']:
                explanation.append("\nThis prediction meets the confidence threshold.")
            else:
                explanation.append("\nWarning: This prediction is below the confidence threshold.")
                
            return "\n".join(explanation)
            
        except Exception as e:
            logging.error(f"Error generating explanation: {e}")
            raise

def main():
    """Main function for testing"""
    try:
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        models_dir = os.path.join(current_dir, 'models')

        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)

        # Check if we need to train a model first
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.model')]
        
        if not model_files:
            logging.info("No existing model found. Training a new model...")
            from xgboost_train_model import train_time_series_model
            
            try:
                # Train the model
                model_path, metrics = train_time_series_model(
                    table_name="strategy_SYM_10021279",  # Replace with your table name
                    target_col="Price",
                    prediction_horizon=1
                )
                logging.info(f"Successfully trained new model: {model_path}")
                logging.info(f"Training metrics: {metrics}")
            except Exception as e:
                logging.error(f"Error training model: {str(e)}")
                raise

        # Initialize predictor
        predictor = ModelPredictor(db_path, models_dir)
        
        if predictor.model is None:
            logging.error("No model available for predictions")
            return

        # Make predictions
        table_name = "strategy_SYM_10021279"  # Replace with your table name
        prediction_result = predictor.make_predictions(
            table_name=table_name,
            n_rows=100,
            confidence_threshold=0.8
        )
        
        # Get and print explanation
        explanation = predictor.get_prediction_explanation(prediction_result)
        print("\nPrediction Explanation:")
        print(explanation)
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()