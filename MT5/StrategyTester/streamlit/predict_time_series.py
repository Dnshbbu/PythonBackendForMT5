import argparse
import logging
import os
import pandas as pd
import json
import numpy as np
import mlflow
import sqlite3
from datetime import datetime
from time_series_predictor import TimeSeriesPredictor
from typing import List, Dict, Optional, Tuple
from model_repository import ModelRepository

# Constants
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'trading_data.db')

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_model_path_from_repository(model_name: str) -> str:
    """Get model path from repository
    
    Args:
        model_name: Name of the model
        
    Returns:
        Path to the model directory
    """
    model_repo = ModelRepository(DB_PATH)
    model_info = model_repo.get_model_info(model_name)
    if not model_info or 'model_path' not in model_info:
        raise ValueError(f"Model not found in repository: {model_name}")
    return model_info['model_path']

def load_data_from_db(db_path: str, table_name: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Load data from SQLite database
    
    Args:
        db_path: Path to SQLite database
        table_name: Name of the table to query
        start_date: Optional start date filter (format: YYYY-MM-DD)
        end_date: Optional end date filter (format: YYYY-MM-DD)
    """
    query = f"SELECT * FROM {table_name}"
    conditions = []
    
    if start_date:
        conditions.append(f"Date >= '{start_date}'")
    if end_date:
        conditions.append(f"Date <= '{end_date}'")
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY Date, Time"  # Ensure data is ordered chronologically
    
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query, conn)
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.set_index('DateTime')
        return df
    finally:
        conn.close()

def save_predictions(predictions: pd.DataFrame, output_path: str, format: str = 'csv'):
    """Save predictions to file
    
    Args:
        predictions: DataFrame containing predictions
        output_path: Path to save the predictions
        format: Output format ('csv' or 'json')
    """
    # Create predictions directory if it doesn't exist
    predictions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    # If output_path is just a filename, save it in predictions directory
    if not os.path.dirname(output_path):
        output_path = os.path.join(predictions_dir, output_path)
    
    if format.lower() == 'csv':
        predictions.to_csv(output_path)
    elif format.lower() == 'json':
        predictions.to_json(output_path, orient='records', date_format='iso')
    else:
        raise ValueError(f"Unsupported output format: {format}")
    
    logging.info(f"Predictions saved to {output_path}")

def store_predictions_in_db(results_df, run_id, model_name, source_table):
    """Store predictions in SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
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
        ''')
        
        # Insert predictions
        for _, row in results_df.iterrows():
            cursor.execute('''
                INSERT INTO historical_predictions 
                (datetime, actual_price, predicted_price, error, price_change, predicted_change, 
                price_volatility, run_id, source_table, model_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row.name.strftime('%Y-%m-%d %H:%M:%S') if isinstance(row.name, datetime) else str(row.name),
                float(row['Actual_Price']),
                float(row['Predicted_Price']),
                float(row['Error']) if 'Error' in row else row['Actual_Price'] - row['Predicted_Price'],
                float(row['Price_Change']) if 'Price_Change' in row else 0.0,
                float(row['Predicted_Change']) if 'Predicted_Change' in row else 0.0,
                float(row['Price_Volatility']) if 'Price_Volatility' in row else 0.0,
                run_id,
                source_table,
                model_name
            ))
        
        conn.commit()
        logging.info(f"Successfully stored {len(results_df)} predictions in database")
    except Exception as e:
        logging.error(f"Error storing predictions: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()

def store_metrics_in_db(metrics, run_id, source_table, model_name, db_path):
    """Store prediction metrics in the database."""
    conn = sqlite3.connect(db_path)
    try:
        # Create metrics table if it doesn't exist
        conn.execute("""
        CREATE TABLE IF NOT EXISTS historical_prediction_metrics (
            id INTEGER PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
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

        # Format current timestamp in a consistent format
        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Insert metrics with explicit timestamp in consistent format
        conn.execute("""
        INSERT INTO historical_prediction_metrics (
            timestamp, run_id, source_table, model_name,
            total_predictions, mean_absolute_error, root_mean_squared_error,
            mean_absolute_percentage_error, r_squared, direction_accuracy,
            up_prediction_accuracy, down_prediction_accuracy, correct_ups,
            correct_downs, total_ups, total_downs, max_error,
            min_error, std_error, avg_price_change, price_volatility,
            mean_prediction_error, median_prediction_error, error_skewness,
            first_quarter_accuracy, last_quarter_accuracy,
            max_correct_streak, avg_correct_streak
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            current_timestamp, run_id, source_table, model_name,
            metrics['total_predictions'], metrics['mean_absolute_error'],
            metrics['root_mean_squared_error'], metrics['mean_absolute_percentage_error'],
            metrics['r_squared'], metrics['direction_accuracy'],
            metrics['up_prediction_accuracy'], metrics['down_prediction_accuracy'],
            metrics['correct_ups'], metrics['correct_downs'],
            metrics['total_ups'], metrics['total_downs'],
            metrics['max_error'], metrics['min_error'],
            metrics['std_error'], metrics['avg_price_change'],
            metrics['price_volatility'], metrics['mean_prediction_error'],
            metrics['median_prediction_error'], metrics['error_skewness'],
            metrics['first_quarter_accuracy'], metrics['last_quarter_accuracy'],
            metrics['max_correct_streak'], metrics['avg_correct_streak']
        ))
        
        conn.commit()
        logging.info("Successfully stored metrics in database")
    except Exception as e:
        logging.error(f"Error storing metrics: {str(e)}")
        raise
    finally:
        conn.close()

def calculate_metrics(results_df: pd.DataFrame) -> Dict:
    """Calculate comprehensive prediction metrics
    
    Args:
        results_df: DataFrame containing actual and predicted values
        
    Returns:
        Dictionary containing various metrics
    """
    try:
        if results_df.empty or len(results_df) < 2:
            logging.warning("Insufficient data for calculating metrics")
            return {}
        
        # Calculate price changes and errors
        results_df['Price_Change'] = results_df['Actual_Price'].diff()
        results_df['Predicted_Change'] = results_df['Predicted_Price'].diff()
        results_df['Error'] = results_df['Actual_Price'] - results_df['Predicted_Price']
        
        # Direction calculations
        correct_direction = (
            (results_df['Price_Change'] > 0) & (results_df['Predicted_Change'] > 0) |
            (results_df['Price_Change'] < 0) & (results_df['Predicted_Change'] < 0)
        )
        
        # Calculate metrics
        total_predictions = len(results_df) - 1  # Subtract 1 because of diff()
        correct_ups = sum((results_df['Price_Change'] > 0) & (results_df['Predicted_Change'] > 0))
        correct_downs = sum((results_df['Price_Change'] < 0) & (results_df['Predicted_Change'] < 0))
        total_ups = sum(results_df['Price_Change'] > 0)
        total_downs = sum(results_df['Price_Change'] < 0)
        
        # Calculate R-squared
        ss_res = sum(results_df['Error'] ** 2)
        ss_tot = sum((results_df['Actual_Price'] - results_df['Actual_Price'].mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Calculate MAPE safely
        mape = np.mean(np.abs(results_df['Error'] / results_df['Actual_Price'])) * 100
        
        # Calculate streaks
        current_streak = max_streak = 0
        streaks = []
        for correct in correct_direction:
            if correct:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            streaks.append(current_streak)
        
        # Price volatility
        results_df['Price_Volatility'] = results_df['Price_Change'].rolling(window=20).std()
        
        metrics = {
            'total_predictions': total_predictions,
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
            'avg_price_change': float(results_df['Price_Change'].mean()),
            'price_volatility': float(results_df['Price_Volatility'].mean()),
            'mean_prediction_error': float(results_df['Error'].mean()),
            'median_prediction_error': float(results_df['Error'].median()),
            'error_skewness': float(results_df['Error'].skew()),
            'first_quarter_accuracy': float(results_df[:len(results_df)//4]['Error'].abs().mean()),
            'last_quarter_accuracy': float(results_df[3*len(results_df)//4:]['Error'].abs().mean()),
            'max_correct_streak': int(max_streak),
            'avg_correct_streak': float(np.mean(streaks) if streaks else 0)
        }
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        raise

def make_predictions(data: pd.DataFrame, predictor: TimeSeriesPredictor) -> Tuple[pd.DataFrame, Dict]:
    """Make predictions using the current row to predict next row's price
    
    Args:
        data: Input DataFrame
        predictor: TimeSeriesPredictor instance
        
    Returns:
        Tuple of (predictions DataFrame, metrics dictionary)
    """
    try:
        logging.info("Preparing data for predictions...")
        
        # Initialize results DataFrame with the same index as the input data
        results_df = pd.DataFrame(index=data.index[predictor.n_lags:])
        results_df['Actual_Price'] = data['Price'][predictor.n_lags:]
        
        # Make predictions
        logging.info("Making predictions...")
        predictions = []
        
        if predictor.model_type in ['SARIMA', 'ARIMA']:
            # For SARIMA/ARIMA models, do one-step-ahead predictions
            for i in range(len(data) - predictor.n_lags):
                # Get the current window of data
                current_window = data['Price'].values[i:i+predictor.n_lags]
                
                # For SARIMA/ARIMA models, we need to:
                # 1. Apply the model to the current window
                # 2. Make a one-step forecast
                if hasattr(predictor.model, 'apply'):
                    # Apply model to current window and get forecast
                    model_fit = predictor.model.apply(current_window)
                    forecast = model_fit.forecast(steps=1)
                else:
                    # If apply is not available, use the base model for prediction
                    forecast = predictor.model.forecast(steps=1)
                
                # Extract prediction value
                pred_value = forecast.iloc[0] if isinstance(forecast, (pd.Series, pd.DataFrame)) else forecast[0]
                predictions.append(float(pred_value))
        else:
            # For other models, make predictions one at a time using sliding windows
            for i in range(len(data) - predictor.n_lags):
                # Get current window of features
                current_window = predictor.prepare_data(data.iloc[i:i+predictor.n_lags+1])
                
                # Make prediction for this window
                pred, _ = predictor.predict(current_window)
                predictions.append(float(pred[0]))
            
        results_df['Predicted_Price'] = predictions
        
        # Calculate metrics
        logging.info("Calculating prediction metrics...")
        metrics = calculate_metrics(results_df)
        
        return results_df, metrics
        
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        raise

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Make predictions using trained time series models')
    
    # Required arguments
    parser.add_argument('--model-name', required=True,
                      help='Name of the trained model')
    parser.add_argument('--table', required=True,
                      help='Name of the table to use for prediction')
    
    # Optional arguments
    parser.add_argument('--start-date',
                      help='Start date for prediction (YYYY-MM-DD)')
    parser.add_argument('--end-date',
                      help='End date for prediction (YYYY-MM-DD)')
    parser.add_argument('--output-path',
                      help='Path to save predictions (default: predictions_{model_name}_{timestamp}.csv)')
    parser.add_argument('--output-format', choices=['csv', 'json'], default='csv',
                      help='Output format for predictions (default: csv)')
    parser.add_argument('--show-metrics', action='store_true',
                      help='Show model metrics')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    setup_logging()
    
    try:
        logging.info(f"Starting prediction process for table: {args.table}")
        logging.info(f"Using model: {args.model_name}")
        
        # Get model path from repository
        model_path = get_model_path_from_repository(args.model_name)
        logging.info(f"Model path: {model_path}")
        
        # Load data from database
        logging.info("Loading data from database...")
        data = load_data_from_db(
            DB_PATH,
            args.table,
            start_date=args.start_date,
            end_date=args.end_date
        )
        logging.info(f"Loaded {len(data)} rows of data")
        
        # Set up MLflow
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mlflow_db = os.path.join(current_dir, "mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
        
        # Set up MLflow experiment
        experiment = mlflow.get_experiment_by_name("time_series_predictions")
        if experiment is None:
            experiment_id = mlflow.create_experiment("time_series_predictions")
        else:
            experiment_id = experiment.experiment_id
        
        mlflow.set_experiment("time_series_predictions")
        
        # Create run_id in the required format: run_YYYYMMDD_HHMMSS_fff
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]  # Include milliseconds but truncate to 3 digits
        run_id = f"run_{current_time}"
        
        with mlflow.start_run(run_name=run_id) as run:
            # Use our run_id format instead of MLflow's internal run_id
            logging.info(f"Using MLflow run: {run_id}")
            
            # Log parameters
            mlflow.log_param("model_name", args.model_name)
            mlflow.log_param("table", args.table)
            mlflow.log_param("data_points", len(data))
            
            # Initialize predictor
            predictor = TimeSeriesPredictor(model_path)
            
            # Make predictions
            results_df, metrics = make_predictions(data, predictor)
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)
            
            # Store predictions and metrics in database using our run_id format
            store_predictions_in_db(results_df, run_id, args.model_name.replace("_sarima", ""), args.table)
            store_metrics_in_db(metrics, run_id, args.table, args.model_name.replace("_sarima", ""), DB_PATH)
            
            # Save predictions to file if requested
            if args.output_path:
                save_predictions(results_df, args.output_path, args.output_format)
            
            # Show metrics if requested
            if args.show_metrics:
                print("\nPrediction Metrics:")
                for metric, value in metrics.items():
                    print(f"{metric}: {value}")
            
            logging.info("Prediction process completed successfully")
            
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main() 