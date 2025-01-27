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
        
        # Load the latest model and scaler
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.load_latest_model()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        


    # def load_latest_model(self) -> None:
    #     """Load the most recent model and its associated metadata"""
    #     try:
    #         # Check if models directory exists
    #         if not os.path.exists(self.models_dir):
    #             logging.warning(f"Models directory not found: {self.models_dir}")
    #             os.makedirs(self.models_dir)
    #             raise FileNotFoundError("Models directory was created but no models found")

    #         # Find all model files
    #         model_files = []
    #         for f in os.listdir(self.models_dir):
    #             if f.endswith('.joblib') and 'scaler' not in f:
    #                 full_path = os.path.join(self.models_dir, f)
    #                 model_files.append((full_path, os.path.getctime(full_path)))

    #         if not model_files:
    #             raise FileNotFoundError("No .joblib model files found in models directory")

    #         # Sort by creation time and get the latest
    #         latest_model_path = sorted(model_files, key=lambda x: x[1], reverse=True)[0][0]
    #         self.current_model_name = os.path.basename(latest_model_path)
    #         base_name = os.path.splitext(os.path.basename(latest_model_path))[0]
            
    #         # Load model
    #         # try:
    #         #     self.model = joblib.load(latest_model_path)
    #         #     logging.info(f"Successfully loaded model from {latest_model_path}")
    #         # except Exception as e:
    #         #     logging.error(f"Failed to load model from {latest_model_path}: {str(e)}")
    #         #     raise
    #         try:
    #             self.model = joblib.load(latest_model_path)
    #             logging.info(f"Successfully loaded model: {self.current_model_name}")
    #         except Exception as e:
    #             logging.error(f"Failed to load model from {latest_model_path}: {str(e)}")
    #             raise

    #         # Try to load feature names from JSON first
    #         feature_names_path = os.path.join(self.models_dir, f"{base_name}_feature_names.json")
    #         if os.path.exists(feature_names_path):
    #             try:
    #                 with open(feature_names_path, 'r') as f:
    #                     feature_data = json.load(f)
    #                 self.feature_columns = feature_data['feature_names']
    #                 logging.info(f"Loaded feature names from JSON: {len(self.feature_columns)} features")
    #             except Exception as e:
    #                 logging.warning(f"Error loading feature names JSON: {str(e)}")
    #                 self.feature_columns = None

    #         # If JSON load failed, try feature importance file
    #         if not self.feature_columns:
    #             importance_path = os.path.join(self.models_dir, f"{base_name}_feature_importance.csv")
    #             if os.path.exists(importance_path):
    #                 try:
    #                     importance_df = pd.read_csv(importance_path)
    #                     self.feature_columns = importance_df['feature'].tolist()
    #                     logging.info(f"Loaded feature names from CSV: {len(self.feature_columns)} features")
    #                 except Exception as e:
    #                     logging.warning(f"Error loading feature importance file: {str(e)}")
    #                     self.feature_columns = None

    #         # If both failed, try to get from model
    #         if not self.feature_columns:
    #             if hasattr(self.model, 'feature_names_'):
    #                 self.feature_columns = self.model.feature_names_
    #                 logging.info("Using feature names from model")
    #             else:
    #                 logging.warning("Using default feature list")
    #                 self.feature_columns = ['Price', 'Score', 'ExitScore']

    #         # Load scaler if available
    #         scaler_path = os.path.join(self.models_dir, f"{base_name}_scaler.joblib")
    #         if os.path.exists(scaler_path):
    #             try:
    #                 self.scaler = joblib.load(scaler_path)
    #                 logging.info("Successfully loaded scaler")
    #             except Exception as e:
    #                 logging.warning(f"Error loading scaler: {str(e)}")
    #                 self.scaler = None
    #         else:
    #             logging.warning("Scaler file not found, will use unscaled features")
    #             self.scaler = None

    #         logging.info(f"Model loading complete. Features: {self.feature_columns}")

    #     except Exception as e:
    #         logging.error(f"Error in load_latest_model: {str(e)}")
    #         self.model = None
    #         self.current_model_name = None
    #         self.feature_columns = ['Price', 'Score', 'ExitScore']
    #         self.scaler = None
    #         raise

    def load_latest_model(self) -> None:
            """Load the most recent model and its associated metadata"""
            try:
                # Check if models directory exists
                if not os.path.exists(self.models_dir):
                    logging.warning(f"Models directory not found: {self.models_dir}")
                    os.makedirs(self.models_dir)
                    raise FileNotFoundError("Models directory was created but no models found")

                # Find all model files
                model_files = []
                for f in os.listdir(self.models_dir):
                    if f.endswith('.joblib') and 'scaler' not in f:
                        full_path = os.path.join(self.models_dir, f)
                        model_files.append((full_path, os.path.getctime(full_path)))

                if not model_files:
                    raise FileNotFoundError("No .joblib model files found in models directory")

                # Sort by creation time and get the latest
                latest_model_path = sorted(model_files, key=lambda x: x[1], reverse=True)[0][0]
                self.current_model_name = os.path.basename(latest_model_path)
                base_name = os.path.splitext(os.path.basename(latest_model_path))[0]
                
                # Load model
                try:
                    self.model = joblib.load(latest_model_path)
                    logging.info(f"Successfully loaded model: {self.current_model_name}")
                except Exception as e:
                    logging.error(f"Failed to load model from {latest_model_path}: {str(e)}")
                    raise

                # Try to load feature names from JSON first
                feature_names_path = os.path.join(self.models_dir, f"{base_name}_feature_names.json")
                if os.path.exists(feature_names_path):
                    try:
                        with open(feature_names_path, 'r') as f:
                            feature_data = json.load(f)
                        self.feature_columns = feature_data['feature_names']
                        logging.info(f"Loaded feature names from JSON: {len(self.feature_columns)} features")
                    except Exception as e:
                        logging.warning(f"Error loading feature names JSON: {str(e)}")
                        self.feature_columns = None

                # If JSON load failed, try feature importance file
                if not self.feature_columns:
                    importance_path = os.path.join(self.models_dir, f"{base_name}_feature_importance.csv")
                    if os.path.exists(importance_path):
                        try:
                            importance_df = pd.read_csv(importance_path)
                            self.feature_columns = importance_df['feature'].tolist()
                            logging.info(f"Loaded feature names from CSV: {len(self.feature_columns)} features")
                        except Exception as e:
                            logging.warning(f"Error loading feature importance file: {str(e)}")
                            self.feature_columns = None

                # If both failed, try to get from model
                if not self.feature_columns:
                    if hasattr(self.model, 'feature_names_'):
                        self.feature_columns = self.model.feature_names_
                        logging.info("Using feature names from model")
                    else:
                        logging.warning("Using default feature list")
                        self.feature_columns = ['Price', 'Score', 'ExitScore']

                # Load scaler if available
                scaler_path = os.path.join(self.models_dir, f"{base_name}_scaler.joblib")
                if os.path.exists(scaler_path):
                    try:
                        self.scaler = joblib.load(scaler_path)
                        logging.info("Successfully loaded scaler")
                    except Exception as e:
                        logging.warning(f"Error loading scaler: {str(e)}")
                        self.scaler = None
                else:
                    logging.warning("Scaler file not found, will use unscaled features")
                    self.scaler = None

                logging.info(f"Model loading complete. Features: {self.feature_columns}")

            except Exception as e:
                logging.error(f"Error in load_latest_model: {str(e)}")
                self.model = None
                self.current_model_name = None
                self.feature_columns = ['Price', 'Score', 'ExitScore']
                self.scaler = None
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
            
            # Make prediction
            prediction = self.model.predict(X.iloc[-1:])
            
            # Get feature importance for this prediction
            feature_importance = dict(zip(X.columns, self.model.feature_importances_))
            top_features = dict(sorted(feature_importance.items(), 
                                    key=lambda x: abs(x[1]), 
                                    reverse=True)[:5])
            
            # Calculate prediction confidence
            confidence = 1.0 - np.std(self.model.predict(X)) / np.mean(np.abs(prediction))
            
            # Prepare result
            result = {
                'timestamp': datetime.now().isoformat(),
                'prediction': float(prediction[0]),
                'confidence': float(confidence),
                'is_confident': confidence >= confidence_threshold,
                'top_features': top_features,
                'model_name': self.current_model_name,
                'metadata': {
                    'features_used': len(self.feature_columns),
                    'data_points': len(df)
                }
            }
            
            # logging.info(f"Made prediction: {result['prediction']:.4f} "
            #             f"(confidence: {result['confidence']:.4f})")
            logging.info(f"Made prediction using model {self.current_model_name}: {result['prediction']:.4f} "
                        f"(confidence: {result['confidence']:.4f})")
            
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

# def main():
#     """Main function for testing"""
#     try:
#         # Setup paths
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
#         models_dir = os.path.join(current_dir, 'models')
        
#         # Initialize predictor
#         predictor = ModelPredictor(db_path, models_dir)
        
#         # Make predictions
#         table_name = "strategy_SYM_10021279"  # Replace with your table name
#         prediction_result = predictor.make_predictions(
#             table_name=table_name,
#             n_rows=100,
#             confidence_threshold=0.8
#         )
        
#         # Get and print explanation
#         explanation = predictor.get_prediction_explanation(prediction_result)
#         print("\nPrediction Explanation:")
#         print(explanation)
        
#     except Exception as e:
#         logging.error(f"Error in main: {e}")
#         raise

# if __name__ == "__main__":
#     main()


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