import os
import logging
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from model_predictor import ModelPredictor
from typing import Dict, List, Optional
import json
import torch
from model_implementations import LSTMModel, TimeSeriesDataset

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

    def store_predictions(self, results_df: pd.DataFrame, summary: Dict, table_name: str) -> None:
        """Store predictions and summary in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create predictions table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_name TEXT,
                    datetime TEXT,
                    actual_price REAL,
                    predicted_price REAL,
                    error REAL,
                    timestamp TEXT,
                    summary_data TEXT,
                    model_name TEXT
                )
            """)
            
            # Convert summary to JSON string
            summary_json = json.dumps(summary)
            
            # Prepare data for insertion
            prediction_data = []
            for idx, row in results_df.iterrows():
                prediction_data.append((
                    table_name,
                    str(idx),  # datetime
                    float(row['Actual_Price']),
                    float(row['Predicted_Price']),
                    float(row['Error']),
                    datetime.now().isoformat(),
                    summary_json,
                    self.model_predictor.current_model_name if self.model_predictor else None
                ))
            
            # Insert predictions
            cursor.executemany("""
                INSERT INTO model_predictions 
                (table_name, datetime, actual_price, predicted_price, error, timestamp, summary_data, model_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
        """Run predictions on historical data"""
        try:
            # Get data from database
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            conn.close()
            
            # Convert date and time to datetime
            if 'Date' in df.columns and 'Time' in df.columns:
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                df = df.set_index('DateTime')
            
            # Prepare features
            X = self.model_predictor.prepare_features(df)
            
            predictions = []
            
            # Handle predictions based on model type
            if hasattr(self.model_predictor, 'model_type') and self.model_predictor.model_type == 'lstm':
                sequence_length = self.model_predictor.sequence_length  # Get sequence length from model_predictor
                device = self.model_predictor.model_params['device']
                
                # Create sequences for prediction
                for i in range(sequence_length - 1, len(df) - 1):
                    try:
                        next_row = df.iloc[i+1]
                        sequence = X.iloc[i-sequence_length+1:i+1].values  # Get the sequence
                        
                        # Convert to tensor and add batch dimension
                        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
                        
                        # Make prediction
                        self.model_predictor.model.eval()  # Set model to evaluation mode
                        with torch.no_grad():
                            prediction = self.model_predictor.model(sequence_tensor)
                            prediction = prediction.cpu().numpy()  # Convert prediction back to numpy
                        
                        predictions.append({
                            'DateTime': next_row.name,
                            'Actual_Price': float(next_row['Price']),
                            'Predicted_Price': float(prediction[0][0]),
                            'Error': float(next_row['Price']) - float(prediction[0][0])
                        })
                        
                    except Exception as e:
                        logging.warning(f"Error processing row {i}: {str(e)}")
                        logging.warning(f"Sequence shape: {sequence.shape if 'sequence' in locals() else 'N/A'}")
                        continue
            else:
                # Original prediction logic for traditional models
                for i in range(len(df) - 1):
                    try:
                        next_row = df.iloc[i+1]
                        current_features = X.iloc[i:i+1]
                        prediction = self.model_predictor.model.predict(current_features)
                        
                        predictions.append({
                            'DateTime': next_row.name,
                            'Actual_Price': float(next_row['Price']),
                            'Predicted_Price': float(prediction[0]),
                            'Error': float(next_row['Price']) - float(prediction[0])
                        })
                        
                    except Exception as e:
                        logging.warning(f"Error processing row {i}: {str(e)}")
                        continue

            # Convert results to DataFrame and set index
            results_df = pd.DataFrame(predictions)
            if not results_df.empty:
                results_df.set_index('DateTime', inplace=True)
            else:
                logging.warning("No predictions were generated")
                results_df = pd.DataFrame(columns=['Actual_Price', 'Predicted_Price', 'Error'])
                results_df.index.name = 'DateTime'
            
            return results_df
            
        except Exception as e:
            logging.error(f"Error running predictions: {str(e)}")
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
    """Main function for testing predictions"""
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize predictor
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        models_dir = os.path.join(current_dir, 'models')
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        # Example of using a specific model
        model_name = "xgboost_single_20250205_151017"  # Replace with None to use latest model
        predictor = HistoricalPredictor(db_path, models_dir, model_name)
        
        # Run predictions for a specific table
        table_name = "strategy_TRIP_NAS_10016827"  # Replace with your table name
        results_df = predictor.run_predictions(table_name)
        
        # Generate and store summary
        summary = predictor.generate_summary(results_df)
        predictor.store_predictions(results_df, summary, table_name)
        
        # Print summary
        print("\nPrediction Summary:")
        print(f"Total predictions: {summary['total_predictions']}")
        print(f"Using model: {predictor.model_predictor.current_model_name}")
        print()
        print("Error Metrics:")
        print(f"Mean absolute error: {summary['mean_absolute_error']:.4f}")
        print(f"Root mean squared error: {summary['root_mean_squared_error']:.4f}")
        print(f"Mean absolute percentage error: {summary['mean_absolute_percentage_error']:.2f}%")
        print(f"R-squared score: {summary['r_squared']:.4f}")
        print()
        print("Error Distribution:")
        print(f"Mean prediction error: {summary['mean_prediction_error']:.4f}")
        print(f"Median prediction error: {summary['median_prediction_error']:.4f}")
        print(f"Error skewness: {summary['error_skewness']:.4f}")
        print(f"Error quartiles: Q25={summary['error_quartiles']['q25']:.4f}, "
              f"Q50={summary['error_quartiles']['q50']:.4f}, "
              f"Q75={summary['error_quartiles']['q75']:.4f}")
        print()
        print("Price Movement Analysis:")
        print(f"Average price change: {summary['avg_price_change']:.4f}")
        print(f"Price volatility: {summary['price_volatility']:.4f}")
        
        # Optional metrics if available
        if 'max_correct_streak' in summary:
            print(f"\nStreak Analysis:")
            print(f"Maximum correct streak: {summary['max_correct_streak']}")
            print(f"Average correct streak: {summary['avg_correct_streak']:.2f}")
        
        if 'direction_accuracy' in summary:
            print(f"\nDirection Analysis:")
            print(f"Direction accuracy: {summary['direction_accuracy']:.2%}")
            print(f"Up prediction accuracy: {summary['up_prediction_accuracy']:.2%}")
            print(f"Down prediction accuracy: {summary['down_prediction_accuracy']:.2%}")
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
