import os
import logging
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import mlflow
import torch
from model_predictor import ModelPredictor
from typing import Dict, List, Optional
import json
import argparse
from model_implementations import LSTMModel, TimeSeriesDataset

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class HistoricalPredictor:
    def __init__(self, db_path: str, models_dir: str, model_name: Optional[str] = None):
        """
        Initialize the predictor with database and model paths
        
        Args:
            db_path: Path to the database
            models_dir: Directory containing models
            model_name: Optional specific model to use
        """
        self.db_path = db_path
        self.models_dir = models_dir
        self.model_predictor = ModelPredictor(db_path, models_dir)
        
        # Load specific model if provided
        if model_name:
            self.model_predictor.load_model_by_name(model_name)
        else:
            self.model_predictor.load_latest_model()
            
        self.setup_logging()
        self.setup_prediction_database()

    def setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def setup_prediction_database(self):
        """Setup SQLite database for predictions and metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create predictions table if it doesn't exist (removed DROP TABLE statements)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    datetime TIMESTAMP,
                    actual_price REAL,
                    predicted_price REAL,
                    error REAL,
                    price_change REAL,
                    predicted_change REAL,
                    price_volatility REAL,
                    run_id TEXT,
                    source_table TEXT,
                    model_name TEXT
                )
            """)
            
            # Create metrics table if it doesn't exist (removed DROP TABLE statements)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_prediction_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    run_id TEXT,
                    source_table TEXT,
                    model_name TEXT,
                    total_predictions INTEGER,
                    mean_absolute_error REAL,
                    root_mean_squared_error REAL,
                    mean_absolute_percentage_error REAL,
                    r_squared REAL,
                    direction_accuracy REAL,
                    up_prediction_accuracy REAL,
                    down_prediction_accuracy REAL,
                    correct_ups INTEGER,
                    correct_downs INTEGER,
                    total_ups INTEGER,
                    total_downs INTEGER,
                    max_error REAL,
                    min_error REAL,
                    std_error REAL,
                    avg_price_change REAL,
                    price_volatility REAL,
                    mean_prediction_error REAL,
                    median_prediction_error REAL,
                    error_skewness REAL,
                    first_quarter_accuracy REAL,
                    last_quarter_accuracy REAL,
                    max_correct_streak INTEGER,
                    avg_correct_streak REAL
                )
            """)
            
            conn.commit()
        except Exception as e:
            logging.error(f"Error setting up database: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def load_data(self, table_name: str) -> pd.DataFrame:
        """Load data from the database table"""
        try:
            query = f"SELECT * FROM {table_name} ORDER BY Date, Time"
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn)
            
            # Convert Date and Time to datetime
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df = df.set_index('DateTime')
            
            # Convert Price to float
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
            
            # Convert numeric columns to float
            numeric_columns = [
                col for col in df.columns 
                if any(prefix in col for prefix in ['Factors_', 'EntryScore_', 'ExitScore'])
            ]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill any NaN values with 0
            df = df.fillna(0)
            
            logging.info(f"Loaded {len(df)} rows from {table_name}")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def store_predictions(self, results_df: pd.DataFrame, summary: Dict, table_name: str, run_id: str) -> None:
        """
        Store predictions and summary in database
        
        Args:
            results_df: DataFrame containing predictions
            summary: Dictionary of summary metrics
            table_name: Name of the source table
            run_id: MLflow run ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prepare data for insertion
            prediction_data = []
            for idx, row in results_df.iterrows():
                # Calculate price volatility if not already present
                price_volatility = row.get('Price_Volatility', 
                    results_df['Actual_Price'].rolling(window=20).std().loc[idx]
                    if len(results_df) >= 20 else 0.0)
                
                prediction_data.append((
                    str(idx),  # datetime
                    float(row['Actual_Price']),
                    float(row['Predicted_Price']),
                    float(row['Error']),
                    float(row.get('Price_Change', 0.0)),
                    float(row.get('Predicted_Change', 0.0)),
                    float(price_volatility),
                    run_id,  # MLflow run_id
                    table_name,  # source_table
                    self.model_predictor.current_model_name if self.model_predictor else None
                ))
            
            # Insert predictions
            cursor.executemany("""
                INSERT INTO historical_predictions 
                (datetime, actual_price, predicted_price, error, 
                price_change, predicted_change, price_volatility,
                run_id, source_table, model_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, prediction_data)
            
            conn.commit()
            conn.close()
            logging.info(f"Successfully stored predictions for {table_name}")
            
        except Exception as e:
            logging.error(f"Error storing predictions: {str(e)}")
            raise

    def store_metrics(self, summary: Dict, run_id: str, source_table: str):
        """Store metrics in SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get model name from ModelPredictor - fixed attribute name
            model_name = self.model_predictor.current_model_name
            
            cursor.execute("""
                INSERT INTO historical_prediction_metrics (
                    timestamp, run_id, source_table, model_name, total_predictions,
                    mean_absolute_error, root_mean_squared_error,
                    mean_absolute_percentage_error, r_squared, direction_accuracy,
                    up_prediction_accuracy, down_prediction_accuracy, correct_ups,
                    correct_downs, total_ups, total_downs, max_error, min_error,
                    std_error, avg_price_change, price_volatility,
                    mean_prediction_error, median_prediction_error,
                    error_skewness, first_quarter_accuracy,
                    last_quarter_accuracy, max_correct_streak,
                    avg_correct_streak
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                run_id,
                source_table,
                model_name,
                summary['total_predictions'],
                summary['mean_absolute_error'],
                summary['root_mean_squared_error'],
                summary['mean_absolute_percentage_error'],
                summary['r_squared'],
                summary['direction_accuracy'],
                summary['up_prediction_accuracy'],
                summary['down_prediction_accuracy'],
                summary['correct_ups'],
                summary['correct_downs'],
                summary['total_ups'],
                summary['total_downs'],
                summary['max_error'],
                summary['min_error'],
                summary['std_error'],
                summary['avg_price_change'],
                summary['price_volatility'],
                summary['mean_prediction_error'],
                summary['median_prediction_error'],
                summary['error_skewness'],
                summary['first_quarter_accuracy'],
                summary['last_quarter_accuracy'],
                summary['max_correct_streak'],
                summary['avg_correct_streak']
            ))
            
            conn.commit()
            logging.info("Stored metrics in database")
            
        except Exception as e:
            logging.error(f"Error storing metrics: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def run_predictions(self, table_name: str) -> pd.DataFrame:
        """Run predictions on historical data and store results"""
        try:
            # Load data
            df = self.load_data(table_name)
            
            # Check if we're in a nested run
            active_run = mlflow.active_run()
            if not active_run:
                # Set MLflow tracking URI to use the same database as your command
                current_dir = os.path.dirname(os.path.abspath(__file__))
                mlflow_db = os.path.join(current_dir, "mlflow.db")
                mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
                
                # Set up MLflow experiment for predictions
                experiment = mlflow.get_experiment_by_name("model_predictions")
                if experiment is None:
                    # Create the experiment if it doesn't exist
                    experiment_id = mlflow.create_experiment("model_predictions")
                    logging.info(f"Created new MLflow experiment with ID: {experiment_id}")
                else:
                    logging.info(f"Using existing MLflow experiment with ID: {experiment.experiment_id}")
                
                mlflow.set_experiment("model_predictions")
                
                # Create a run_name in the format we want
                current_time = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]  # Include milliseconds but truncate to 3 digits
                run_name = f"run_{current_time}"
                
                # Start MLflow run with our custom run_name
                run_context = mlflow.start_run(run_name=run_name)
            else:
                run_context = active_run
                run_name = active_run.info.run_name
            
            with run_context:
                run_id = run_name  # Use the same format for run_id
                logging.info(f"Using MLflow run: {run_id}")
                
                # Log model info
                mlflow.log_param("model_name", self.model_predictor.current_model_name)
                mlflow.log_param("source_table", table_name)
                mlflow.log_param("data_points", len(df))
                
                # Get predictions
                results_df = pd.DataFrame()
                results_df['Actual_Price'] = df['Price']
                
                # Prepare features and make predictions
                X = self.model_predictor.prepare_features(df)
                
                if isinstance(self.model_predictor.model, LSTMModel):
                    # Handle LSTM predictions
                    sequence_length = self.model_predictor.sequence_length
                    logging.info(f"Preparing LSTM predictions with sequence length: {sequence_length}")
                    logging.info(f"X shape before dataset creation: {X.shape}")
                    logging.info(f"X type: {type(X)}")
                    
                    # Create dataset - X is already a numpy array from prepare_features
                    dataset = TimeSeriesDataset(X, df['Price'].values, sequence_length)
                    predictions = []
                    
                    with torch.no_grad():
                        for i in range(len(dataset)):
                            x, _ = dataset[i]
                            x = torch.FloatTensor(x).unsqueeze(0)
                            pred = self.model_predictor.model(x)
                            predictions.append(pred.item())
                    
                    logging.info(f"DataFrame length: {len(df)}")
                    logging.info(f"Predictions length: {len(predictions)}")
                    logging.info(f"Sequence length: {sequence_length}")
                    
                    # Pad the beginning with NaN values due to sequence length
                    pad = [np.nan] * (sequence_length - 1)
                    # Ensure predictions array matches DataFrame length
                    if len(pad) + len(predictions) < len(df):
                        # Add one more NaN if needed
                        pad = [np.nan] * sequence_length
                        logging.info("Added extra padding to match DataFrame length")
                    elif len(pad) + len(predictions) > len(df):
                        # Trim predictions if needed
                        predictions = predictions[:len(df) - len(pad)]
                        logging.info("Trimmed predictions to match DataFrame length")
                    
                    logging.info(f"Final combined length: {len(pad) + len(predictions)}")
                    results_df['Predicted_Price'] = pad + predictions
                else:
                    # Handle other model types
                    results_df['Predicted_Price'] = self.model_predictor.model.predict(X)
                
                # Calculate prediction errors and changes
                results_df['Error'] = results_df['Actual_Price'] - results_df['Predicted_Price']
                results_df['Price_Change'] = results_df['Actual_Price'].diff()
                results_df['Predicted_Change'] = results_df['Predicted_Price'].diff()
                
                # Generate summary metrics
                summary = self.generate_summary(results_df)
                
                # Log metrics to MLflow
                for metric_name, metric_value in summary.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)
                
                # Save predictions DataFrame as artifact
                temp_csv = "temp_predictions.csv"
                results_df.to_csv(temp_csv)
                mlflow.log_artifact(temp_csv)
                os.remove(temp_csv)
                
                # Store predictions and metrics in SQLite with the same run_id
                self.store_predictions(results_df, summary, table_name, run_id)
                self.store_metrics(summary, run_id, table_name)
                
                logging.info(f"Successfully ran predictions on {table_name}")
                return results_df
                
        except Exception as e:
            logging.error(f"Error running predictions: {e}")
            raise

    def generate_summary(self, results_df: pd.DataFrame) -> Dict:
        """Generate comprehensive summary statistics for predictions"""
        try:
            # Check if DataFrame is empty or has insufficient data
            if results_df.empty or len(results_df) < 2:
                logging.warning("Insufficient data for generating summary")
                return {
                    'total_predictions': 0,
                    'mean_absolute_error': 0.0,
                    'root_mean_squared_error': 0.0,
                    'mean_absolute_percentage_error': 0.0,
                    'r_squared': 0.0,
                    'direction_accuracy': 0.0,
                    'up_prediction_accuracy': 0.0,
                    'down_prediction_accuracy': 0.0,
                    'correct_ups': 0,
                    'correct_downs': 0,
                    'total_ups': 0,
                    'total_downs': 0,
                    'max_error': 0.0,
                    'min_error': 0.0,
                    'std_error': 0.0,
                    'avg_price_change': 0.0,
                    'price_volatility': 0.0,
                    'mean_prediction_error': 0.0,
                    'median_prediction_error': 0.0,
                    'error_skewness': 0.0,
                    'first_quarter_accuracy': 0.0,
                    'last_quarter_accuracy': 0.0,
                    'max_correct_streak': 0,
                    'avg_correct_streak': 0.0,
                    'error_quartiles': {
                        'q25': 0.0,
                        'q50': 0.0,
                        'q75': 0.0
                    }
                }

            # Calculate price changes
            results_df['Price_Change'] = results_df['Actual_Price'].diff()
            results_df['Predicted_Change'] = results_df['Predicted_Price'].diff()
            
            # Direction calculations
            correct_direction = (
                (results_df['Price_Change'] > 0) & (results_df['Predicted_Change'] > 0) |
                (results_df['Price_Change'] < 0) & (results_df['Predicted_Change'] < 0)
            )
            
            # Safe calculations with error handling
            try:
                mape = np.mean(np.abs(results_df['Error'] / results_df['Actual_Price'])) * 100
            except ZeroDivisionError:
                mape = 0.0
            
            try:
                denominator = np.sum((results_df['Actual_Price'] - results_df['Actual_Price'].mean())**2)
                if denominator == 0:
                    r2 = 0.0
                else:
                    r2 = 1 - (np.sum(results_df['Error']**2) / denominator)
            except Exception:
                r2 = 0.0
            
            # Calculate directional metrics with error handling
            total_predictions = len(results_df) - 1  # Subtract 1 because of diff()
            correct_ups = sum((results_df['Price_Change'] > 0) & (results_df['Predicted_Change'] > 0))
            correct_downs = sum((results_df['Price_Change'] < 0) & (results_df['Predicted_Change'] < 0))
            total_ups = sum(results_df['Price_Change'] > 0)
            total_downs = sum(results_df['Price_Change'] < 0)
            
            # Add price movement analysis
            results_df['Price_Volatility'] = results_df['Price_Change'].rolling(window=20).std()
            
            # Calculate additional metrics with safe division
            summary = {
                'total_predictions': len(results_df),
                'mean_absolute_error': float(results_df['Error'].abs().mean()) if not results_df.empty else 0.0,
                'root_mean_squared_error': float(np.sqrt((results_df['Error'] ** 2).mean())) if not results_df.empty else 0.0,
                'mean_absolute_percentage_error': float(mape),
                'r_squared': float(r2),
                'direction_accuracy': float(correct_direction.mean()) if not results_df.empty else 0.0,
                'up_prediction_accuracy': float(correct_ups / total_ups if total_ups > 0 else 0),
                'down_prediction_accuracy': float(correct_downs / total_downs if total_downs > 0 else 0),
                'correct_ups': int(correct_ups),
                'correct_downs': int(correct_downs),
                'total_ups': int(total_ups),
                'total_downs': int(total_downs),
                'max_error': float(results_df['Error'].abs().max()) if not results_df.empty else 0.0,
                'min_error': float(results_df['Error'].abs().min()) if not results_df.empty else 0.0,
                'std_error': float(results_df['Error'].std()) if not results_df.empty else 0.0,
                'timestamp': datetime.now().isoformat(),
                
                # Price movement metrics
                'avg_price_change': float(results_df['Price_Change'].mean()) if not results_df.empty else 0.0,
                'price_volatility': float(results_df['Price_Volatility'].mean()) if not results_df.empty else 0.0,
                
                # Prediction bias metrics
                'mean_prediction_error': float(results_df['Error'].mean()) if not results_df.empty else 0.0,
                'median_prediction_error': float(results_df['Error'].median()) if not results_df.empty else 0.0,
                'error_skewness': float(results_df['Error'].skew()) if not results_df.empty else 0.0,
                
                # Time-based accuracy
                'first_quarter_accuracy': float(results_df[:len(results_df)//4]['Error'].abs().mean()) if not results_df.empty else 0.0,
                'last_quarter_accuracy': float(results_df[3*len(results_df)//4:]['Error'].abs().mean()) if not results_df.empty else 0.0,
                
                # Streak analysis
                'max_correct_streak': self._calculate_max_streak(correct_direction) if not results_df.empty else 0,
                'avg_correct_streak': self._calculate_avg_streak(correct_direction) if not results_df.empty else 0.0,
                
                # Error distribution
                'error_quartiles': {
                    'q25': float(results_df['Error'].quantile(0.25)) if not results_df.empty else 0.0,
                    'q50': float(results_df['Error'].quantile(0.50)) if not results_df.empty else 0.0,
                    'q75': float(results_df['Error'].quantile(0.75)) if not results_df.empty else 0.0
                }
            }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            raise

    def _calculate_max_streak(self, correct_series):
        """Calculate maximum streak of correct predictions"""
        current_streak = max_streak = 0
        for correct in correct_series:
            if correct:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    def _calculate_avg_streak(self, correct_series):
        """Calculate average streak length of correct predictions"""
        streaks = []
        current_streak = 0
        for correct in correct_series:
            if correct:
                current_streak += 1
            elif current_streak > 0:
                streaks.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            streaks.append(current_streak)
        return float(np.mean(streaks)) if streaks else 0.0

def main():
    """Command line interface for running predictions"""
    parser = argparse.ArgumentParser(description='Run predictions using trained models')
    
    # Required arguments
    parser.add_argument('--table', required=True,
                       help='Table name to run predictions on')
    
    # Optional arguments
    parser.add_argument('--model-name',
                       help='Specific model to use (default: latest model)')
    parser.add_argument('--output-format', choices=['db', 'csv', 'both'], default='db',
                       help='Output format for predictions (default: db)')
    parser.add_argument('--output-path',
                       help='Path for CSV output if output-format is csv or both')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for predictions (default: 1000)')
    parser.add_argument('--force', action='store_true',
                       help='Force rerun predictions even if they exist')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level))
    
    try:
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        models_dir = os.path.join(current_dir, 'models')
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize predictor
        predictor = HistoricalPredictor(
            db_path=db_path,
            models_dir=models_dir,
            model_name=args.model_name
        )
        
        # Run predictions
        results_df = predictor.run_predictions(args.table)
        
        # Handle output based on format
        if args.output_format in ['csv', 'both']:
            if not args.output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                args.output_path = f"predictions_{timestamp}.csv"
            results_df.to_csv(args.output_path)
            logging.info(f"Predictions saved to CSV: {args.output_path}")
        
        if args.output_format in ['db', 'both']:
            logging.info("Predictions saved to database")
        
        # Print summary statistics
        print("\nPrediction Summary:")
        print(f"Total predictions: {len(results_df)}")
        print(f"Using model: {predictor.model_predictor.current_model_name}")
        print(f"\nError Metrics:")
        print(f"Mean absolute error: {results_df['Error'].abs().mean():.4f}")
        print(f"Root mean squared error: {np.sqrt((results_df['Error'] ** 2).mean()):.4f}")
        
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main()
