import os
import logging
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from model_predictor import ModelPredictor
from typing import Dict, List, Optional
import json

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

    def store_predictions(self, results_df: pd.DataFrame, run_id: str, source_table: str):
        """Store predictions in SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get model name from ModelPredictor - fixed attribute name
            model_name = self.model_predictor.current_model_name
            
            # Prepare data for insertion
            data_to_insert = []
            for idx, row in results_df.iterrows():
                data_to_insert.append((
                    idx.strftime('%Y-%m-%d %H:%M:%S'),
                    row['Actual_Price'],
                    row['Predicted_Price'],
                    row['Error'],
                    row['Price_Change'],
                    row['Predicted_Change'],
                    row.get('Price_Volatility', 0),
                    run_id,
                    source_table,
                    model_name
                ))
            
            # Insert predictions in batch
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT INTO historical_predictions (
                    datetime, actual_price, predicted_price, error,
                    price_change, predicted_change, price_volatility, run_id,
                    source_table, model_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data_to_insert)
            
            conn.commit()
            logging.info(f"Stored {len(data_to_insert)} predictions in database")
            
        except Exception as e:
            logging.error(f"Error storing predictions: {e}")
            raise
        finally:
            if conn:
                conn.close()

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
            # Load data
            df = self.load_data(table_name)
            
            # Initialize results storage
            predictions = []
            
            # Get required features from model
            required_features = self.model_predictor.feature_columns
            
            # Prepare features for prediction
            X = self.model_predictor.prepare_features(df)
            
            # Run predictions for each row except the last one
            for i in range(len(df)-1):  # Note the -1 here
                try:
                    current_row = df.iloc[i]
                    next_row = df.iloc[i+1]  # Get the next row for actual price
                    current_features = X.iloc[i:i+1]
                    
                    # Make prediction for next price
                    prediction = self.model_predictor.model.predict(current_features)
                    
                    # Store results - compare prediction with next row's price
                    predictions.append({
                        'DateTime': next_row.name,  # Use next row's datetime
                        'Actual_Price': float(next_row['Price']),
                        'Predicted_Price': float(prediction[0]),
                        'Error': float(next_row['Price']) - float(prediction[0])
                    })
                    
                    if i % 100 == 0:
                        logging.info(f"Processed {i}/{len(df)} rows")
                        
                except Exception as e:
                    logging.warning(f"Error processing row {i}: {e}")
                    continue
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(predictions)
            results_df.set_index('DateTime', inplace=True)
            
            return results_df
            
        except Exception as e:
            logging.error(f"Error running predictions: {e}")
            raise

    def generate_summary(self, results_df: pd.DataFrame) -> Dict:
        """Generate comprehensive summary statistics for predictions"""
        try:
            # Calculate price changes
            results_df['Price_Change'] = results_df['Actual_Price'].diff()
            results_df['Predicted_Change'] = results_df['Predicted_Price'].diff()
            
            # Direction calculations
            correct_direction = (
                (results_df['Price_Change'] > 0) & (results_df['Predicted_Change'] > 0) |
                (results_df['Price_Change'] < 0) & (results_df['Predicted_Change'] < 0)
            )
            
            # Calculate various metrics
            mape = np.mean(np.abs(results_df['Error'] / results_df['Actual_Price'])) * 100
            r2 = 1 - (np.sum(results_df['Error']**2) / np.sum((results_df['Actual_Price'] - results_df['Actual_Price'].mean())**2))
            
            # Calculate directional metrics
            total_predictions = len(results_df) - 1  # Subtract 1 because of diff()
            correct_ups = sum((results_df['Price_Change'] > 0) & (results_df['Predicted_Change'] > 0))
            correct_downs = sum((results_df['Price_Change'] < 0) & (results_df['Predicted_Change'] < 0))
            total_ups = sum(results_df['Price_Change'] > 0)
            total_downs = sum(results_df['Price_Change'] < 0)
            
            # Add price movement analysis
            results_df['Price_Volatility'] = results_df['Price_Change'].rolling(window=20).std()
            
            # Calculate additional metrics
            summary = {
                'total_predictions': len(results_df),
                'mean_absolute_error': float(results_df['Error'].abs().mean()),
                'root_mean_squared_error': float(np.sqrt((results_df['Error'] ** 2).mean())),
                'mean_absolute_percentage_error': float(mape),
                'r_squared': float(r2),
                'direction_accuracy': float(correct_direction.mean()),
                'up_prediction_accuracy': float(correct_ups / total_ups if total_ups > 0 else 0),
                'down_prediction_accuracy': float(correct_downs / total_downs if total_downs > 0 else 0),
                'correct_ups': int(correct_ups),
                'correct_downs': int(correct_downs),
                'total_ups': int(total_ups),
                'total_downs': int(total_downs),
                'max_error': float(results_df['Error'].abs().max()),
                'min_error': float(results_df['Error'].abs().min()),
                'std_error': float(results_df['Error'].std()),
                'timestamp': datetime.now().isoformat(),
                
                # Price movement metrics
                'avg_price_change': float(results_df['Price_Change'].mean()),
                'price_volatility': float(results_df['Price_Volatility'].mean()),
                'max_price_change': float(results_df['Price_Change'].abs().max()),
                'min_price_change': float(results_df['Price_Change'].abs().min()),
                
                # Prediction bias metrics
                'mean_prediction_error': float(results_df['Error'].mean()),
                'median_prediction_error': float(results_df['Error'].median()),
                'error_skewness': float(results_df['Error'].skew()),
                
                # Time-based accuracy
                'first_quarter_accuracy': float(results_df[:len(results_df)//4]['Error'].abs().mean()),
                'last_quarter_accuracy': float(results_df[3*len(results_df)//4:]['Error'].abs().mean()),
                
                # Streak analysis
                'max_correct_streak': self._calculate_max_streak(correct_direction),
                'avg_correct_streak': self._calculate_avg_streak(correct_direction),
                
                # Error distribution
                'error_quartiles': {
                    'q25': float(results_df['Error'].quantile(0.25)),
                    'q50': float(results_df['Error'].quantile(0.50)),
                    'q75': float(results_df['Error'].quantile(0.75))
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
    """Main function to run predictions"""
    try:
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        models_dir = os.path.join(current_dir, 'models')
        
        # Optional: Specify model name
        model_name = "random_forest_multi_20250204_162853"  # Replace with your model name or None for latest
        
        # Initialize predictor with optional model name
        predictor = HistoricalPredictor(db_path, models_dir, model_name)
        
        # Run predictions
        # table_name = "strategy_TRIP_NAS_10032544"  # Replace with your table name
        # table_name = "strategy_TRIP_NAS_10016827"  # Replace with your table name

        table_name = "strategy_TRIP_NAS_10027636"  # Replace with your table name


        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        results_df = predictor.run_predictions(table_name)
        summary = predictor.generate_summary(results_df)
        
        # Store results in database with source table
        predictor.store_predictions(results_df, run_id, table_name)
        predictor.store_metrics(summary, run_id, table_name)
        
        # Print summary (keeping the existing print statements)
        print("\nPrediction Summary:")
        print(f"Total predictions: {summary['total_predictions']}")
        
        print("\nError Metrics:")
        print(f"Mean absolute error: {summary['mean_absolute_error']:.4f}")
        print(f"Root mean squared error: {summary['root_mean_squared_error']:.4f}")
        print(f"Mean absolute percentage error: {summary['mean_absolute_percentage_error']:.2f}%")
        print(f"R-squared score: {summary['r_squared']:.4f}")
        
        print("\nError Distribution:")
        print(f"Mean prediction error: {summary['mean_prediction_error']:.4f}")
        print(f"Median prediction error: {summary['median_prediction_error']:.4f}")
        print(f"Error skewness: {summary['error_skewness']:.4f}")
        print(f"Error quartiles: Q25={summary['error_quartiles']['q25']:.4f}, "
              f"Q50={summary['error_quartiles']['q50']:.4f}, "
              f"Q75={summary['error_quartiles']['q75']:.4f}")
        
        print("\nPrice Movement Analysis:")
        print(f"Average price change: {summary['avg_price_change']:.4f}")
        print(f"Price volatility: {summary['price_volatility']:.4f}")
        print(f"Maximum price change: {summary['max_price_change']:.4f}")
        
        print("\nPrediction Streaks:")
        print(f"Maximum correct streak: {summary['max_correct_streak']}")
        print(f"Average correct streak: {summary['avg_correct_streak']:.2f}")
        
        print("\nTime-based Performance:")
        print(f"First quarter MAE: {summary['first_quarter_accuracy']:.4f}")
        print(f"Last quarter MAE: {summary['last_quarter_accuracy']:.4f}")
        
        print("\nDirectional Accuracy:")
        print(f"Overall direction accuracy: {summary['direction_accuracy']*100:.2f}%")
        print(f"Upward movement accuracy: {summary['up_prediction_accuracy']*100:.2f}%")
        print(f"Downward movement accuracy: {summary['down_prediction_accuracy']*100:.2f}%")
        
        print("\nMovement Breakdown:")
        print(f"Correct upward predictions: {summary['correct_ups']}/{summary['total_ups']}")
        print(f"Correct downward predictions: {summary['correct_downs']}/{summary['total_downs']}")
        
        logging.info(f"Completed prediction run: {run_id} for table: {table_name}")
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
