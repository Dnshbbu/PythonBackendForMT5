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
import joblib
from sklearn.preprocessing import StandardScaler

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

def get_model_features(model_name: str) -> Tuple[List[str], List[str]]:
    """Get the features used during model training from metadata.json
    
    Args:
        model_name: Name of the model
        
    Returns:
        Tuple of (base_features, all_features) where:
        - base_features: List of original features without lags
        - all_features: List of all features including lags
    """
    # Get model path
    model_path = get_model_path_from_repository(model_name)
    metadata_path = os.path.join(model_path, 'metadata.json')
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        all_features = metadata['features']
        
        # Get base features from diff_orders as they represent all original features
        diff_orders = metadata['metrics']['diff_orders']
        base_features = [f for f in diff_orders.keys() if not f.endswith(('_lag_1', '_lag_2', '_lag_3'))]
        
        # Sort base_features to match order in all_features
        base_features = sorted(base_features, key=lambda x: all_features.index(x) if x in all_features else len(all_features))
        
        logging.info(f"Loaded features from metadata.json: {len(base_features)} base features, {len(all_features)} total features")
        logging.info(f"Differencing orders loaded for features: {diff_orders}")
        
        return base_features, all_features
    except Exception as e:
        logging.error(f"Error loading features from metadata.json: {str(e)}")
        logging.info("Falling back to database for feature information")
        
        # Fallback to database if metadata.json fails
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT features FROM model_repository WHERE model_name = ?", (model_name,))
            features_str = cursor.fetchone()
            if not features_str or not features_str[0]:
                raise ValueError(f"No features found for model: {model_name}")
                
            all_features = json.loads(features_str[0])
            # Extract base features (non-lagged features)
            base_features = [f for f in all_features if not any(f.endswith(f'_lag_{i}') for i in range(1, 100))]
            
            return base_features, all_features
        finally:
            conn.close()

def make_predictions(data: pd.DataFrame, predictor: TimeSeriesPredictor, model_name: str, forecast_horizon: int = 1) -> Tuple[pd.DataFrame, Dict]:
    """Make predictions using the current row to predict next row's price
    
    Args:
        data: Input DataFrame
        predictor: TimeSeriesPredictor instance
        model_name: Name of the model used for prediction
        forecast_horizon: Number of steps to forecast ahead (default: 1)
        
    Returns:
        Tuple of (predictions DataFrame, metrics dictionary)
    """
    try:
        logging.info("Preparing data for predictions...")
        
        # Get base features and all features used during training
        base_features, all_features = get_model_features(model_name)
        logging.info(f"Using base features from training: {base_features}")
        logging.info(f"Total features including lags: {all_features}")
        
        # Create a copy of the data for predictions
        prediction_data = data.copy()
        
        # Add Price_current if it's a base feature but not in data
        if 'Price_current' in base_features and 'Price_current' not in prediction_data.columns:
            if 'Price' not in prediction_data.columns:
                raise ValueError("'Price' column is required in the input data")
            prediction_data['Price_current'] = prediction_data['Price']
            logging.info("Created 'Price_current' from 'Price' column")
        
        # Verify all required base features are present in the data
        missing_features = [f for f in base_features if f not in prediction_data.columns]
        if missing_features:
            raise ValueError(f"Missing required base features in prediction data: {missing_features}")
        
        # Filter data to only use base features
        prediction_data = prediction_data[base_features].copy()
        logging.info(f"Filtered data to use only base features. Shape: {prediction_data.shape}")
        
        # For non-VAR models, map Price_current back to Price for compatibility
        if predictor.model_type != 'VAR':
            if 'Price_current' in prediction_data.columns and 'Price' not in prediction_data.columns:
                prediction_data['Price'] = prediction_data['Price_current']
                logging.info("Mapped 'Price_current' back to 'Price' for model compatibility")
        
        # Initialize results DataFrame with the same index as the input data
        results_df = pd.DataFrame(index=prediction_data.index[predictor.n_lags:])
        results_df['Actual_Price'] = prediction_data['Price' if 'Price' in prediction_data else 'Price_current'][predictor.n_lags:]
        
        # Make predictions
        logging.info("Making predictions...")
        predictions = []
        
        if predictor.model_type == 'VAR':
            # Create lagged features for all base features if they exist in all_features
            for feature in base_features:
                for i in range(1, predictor.n_lags + 1):
                    lag_feature = f'{feature}_lag_{i}'
                    if lag_feature in all_features:
                        prediction_data[lag_feature] = prediction_data[feature].shift(i)
            
            # Drop rows with NaN from lagged features
            prediction_data = prediction_data.dropna()
            
            # Load scaler
            scaler_path = os.path.join(predictor.model_path, 'scaler.pkl')
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                logging.info("Loaded scaler from: %s", scaler_path)
                logging.info("Scaler feature names: %s", scaler.feature_names_in_.tolist() if hasattr(scaler, 'feature_names_in_') else "Not available")
            else:
                raise ValueError(f"Scaler not found at {scaler_path}")
            
            # Get feature columns in the exact order they were used during training
            feature_cols = scaler.feature_names_in_.tolist() if hasattr(scaler, 'feature_names_in_') else all_features
            logging.info(f"Using features for prediction (in order): {feature_cols}")
            
            # Create a mapping dictionary for feature names
            feature_mapping = {
                'Price': 'Price_current',
                'Price_lag_1': 'Price_current_lag_1',
                'Price_lag_2': 'Price_current_lag_2',
                'Price_lag_3': 'Price_current_lag_3'
            }
            
            # Create mapped features
            for scaler_feature in feature_cols:
                if scaler_feature in feature_mapping:
                    mapped_feature = feature_mapping[scaler_feature]
                    if mapped_feature == 'Price_current':
                        prediction_data['Price'] = prediction_data['Price_current']
                    else:
                        lag_num = int(mapped_feature.split('_')[-1])
                        prediction_data[scaler_feature] = prediction_data['Price_current'].shift(lag_num)
            
            # Ensure all features (including lags) are present
            missing_features = [f for f in feature_cols if f not in prediction_data.columns]
            if missing_features:
                raise ValueError(f"Missing features after lag creation: {missing_features}")
            
            # Get numeric data only, in the exact order of training features
            numeric_data = prediction_data[feature_cols]
            
            # Get differencing orders from metadata
            diff_orders = predictor.metadata.get('diff_orders', {})
            logging.info(f"Differencing orders: {diff_orders}")
            
            # Make predictions one step at a time
            for i in range(len(prediction_data) - predictor.n_lags):
                try:
                    # Get current window
                    current_data = numeric_data.iloc[i:i+predictor.n_lags+1]
                    
                    # Apply differencing if needed
                    diff_data = current_data.copy()
                    for column in diff_data.columns:
                        if column in diff_orders:
                            for _ in range(diff_orders[column]):
                                diff_data[column] = np.diff(diff_data[column], prepend=diff_data[column].iloc[0])
                    
                    # Scale the data
                    scaled_data = scaler.transform(diff_data)
                    
                    # Get last observation for prediction
                    last_observation = scaled_data[-1:].reshape(1, -1)
                    
                    # Make prediction
                    forecast = predictor.model.forecast(last_observation, steps=1)
                    
                    # Inverse transform the prediction
                    forecast_unscaled = scaler.inverse_transform(forecast)
                    
                    # Inverse differencing if needed
                    for col_idx, column in enumerate(numeric_data.columns):
                        if column in diff_orders:
                            for _ in range(diff_orders[column]):
                                forecast_unscaled[0, col_idx] = forecast_unscaled[0, col_idx] + current_data[column].iloc[0]
                    
                    # Get the price prediction (should be the first column as per training)
                    price_idx = feature_cols.index('Price') if 'Price' in feature_cols else 0
                    predictions.append(float(forecast_unscaled[0, price_idx]))
                    
                except Exception as e:
                    logging.error(f"Error in VAR prediction step {i}: {str(e)}")
                    raise
        else:
            # Handle other model types
            for i in range(len(prediction_data) - predictor.n_lags):
                # Get current window of features
                current_window = predictor.prepare_data(prediction_data.iloc[i:i+predictor.n_lags+1])
                
                # Make prediction for this window
                pred, _ = predictor.predict(current_window)
                predictions.append(float(pred[0]))
        
        # Store predictions in results DataFrame
        results_df['Predicted_Price'] = predictions
        
        # Calculate metrics
        logging.info("Calculating prediction metrics...")
        metrics = calculate_metrics(results_df)
        
        return results_df, metrics
        
    except Exception as e:
        logging.error(f"Error making predictions: {str(e)}")
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
    parser.add_argument('--forecast-horizon', type=int, default=1,
                      help='Number of steps to forecast ahead (default: 1)')
    
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
        
        # Set up MLflow with absolute path
        mlflow_db = "C:\\Users\\StdUser\\Desktop\\MyProjects\\Backtesting\\MT5\\StrategyTester\\streamlit\\mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
        
        # Set up MLflow experiment
        experiment = mlflow.get_experiment_by_name("model_predictions")
        if experiment is None:
            experiment_id = mlflow.create_experiment("model_predictions")
        else:
            experiment_id = experiment.experiment_id
        
        mlflow.set_experiment("model_predictions")
        
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
            mlflow.log_param("source_table", args.table)
            
            # Initialize predictor
            predictor = TimeSeriesPredictor(model_path)
            
            # Log target column from predictor metadata
            mlflow.log_param("target_col", predictor.target if hasattr(predictor, 'target') else None)
            
            # Make predictions - pass model_name as argument
            results_df, metrics = make_predictions(data, predictor, args.model_name, args.forecast_horizon)
            
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