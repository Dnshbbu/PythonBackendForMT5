import argparse
import logging
import os
import pandas as pd
import json
from datetime import datetime
from time_series_predictor import TimeSeriesPredictor
from typing import List, Dict

# Constants
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'trading_data.db')

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_data_from_db(db_path: str, table_name: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Load data from SQLite database
    
    Args:
        db_path: Path to SQLite database
        table_name: Name of the table to query
        start_date: Optional start date filter (format: YYYY-MM-DD)
        end_date: Optional end date filter (format: YYYY-MM-DD)
    """
    import sqlite3
    
    query = f"SELECT * FROM {table_name}"
    conditions = []
    
    if start_date:
        conditions.append(f"Date >= '{start_date}'")
    if end_date:
        conditions.append(f"Date <= '{end_date}'")
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Make predictions using trained time series models')
    
    # Required arguments
    parser.add_argument('--model-path', required=True,
                      help='Path to the trained model directory')
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
                      help='Number of future time points to predict (default: 1)')
    
    return parser.parse_args()

def format_predictions(data: pd.DataFrame, predictions: List[float], prediction_info: Dict) -> pd.DataFrame:
    """Format predictions into a DataFrame
    
    Args:
        data: Original data used for prediction
        predictions: List of predicted values
        prediction_info: Dictionary containing prediction info
        
    Returns:
        DataFrame containing predictions and confidence intervals
    """
    # Create future dates based on the last date in the data
    freq = pd.infer_freq(data.index)
    if freq is None:
        freq = 'T'  # Default to minutes if frequency cannot be inferred
    future_dates = pd.date_range(start=data.index[-1], periods=len(predictions) + 1, freq=freq)[1:]
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'DateTime': future_dates,
        'Actual': [None] * len(predictions),  # No actual values for future predictions
        'Predicted': predictions,
        'Lower_Bound': prediction_info.get('lower_bound', [None] * len(predictions)),
        'Upper_Bound': prediction_info.get('upper_bound', [None] * len(predictions))
    })
    
    predictions_df.set_index('DateTime', inplace=True)
    return predictions_df

def main():
    """Main function"""
    args = parse_args()
    setup_logging()  # Ensure logging is set up
    
    try:
        logging.info(f"Starting prediction process for table: {args.table}")
        logging.info(f"Using model from: {args.model_path}")
        
        # Load data from database
        logging.info("Loading data from database...")
        data = load_data_from_db(
            DB_PATH,
            args.table,
            start_date=args.start_date,
            end_date=args.end_date
        )
        logging.info(f"Loaded {len(data)} rows of data")
        
        # Initialize predictor
        logging.info("Initializing predictor...")
        predictor = TimeSeriesPredictor(args.model_path)
        predictor.forecast_horizon = args.forecast_horizon
        logging.info(f"Forecast horizon set to: {args.forecast_horizon}")
        
        # Show model info if requested
        if args.show_metrics:
            logging.info("Model Information:")
            model_info = predictor.get_model_info()
            logging.info(f"Model type: {model_info['model_type']}")
            logging.info(f"Target variable: {model_info['target']}")
            logging.info(f"Features: {', '.join(model_info['features'])}")
            logging.info(f"Number of lags: {model_info['n_lags']}")
            
            # Show metrics if available
            if 'metrics' in model_info:
                logging.info("Model Metrics:")
                for metric, value in model_info['metrics'].items():
                    logging.info(f"{metric}: {value:.4f}")
        
        # Prepare data and make predictions
        logging.info("Preparing data for prediction...")
        prepared_data = predictor.prepare_data(data)
        logging.info(f"Data prepared successfully. Shape: {prepared_data.shape}")
        
        logging.info("Making predictions...")
        predictions, prediction_info = predictor.predict(prepared_data)
        logging.info(f"Generated {len(predictions)} predictions")
        
        # Format predictions
        logging.info("Formatting predictions...")
        results = format_predictions(prepared_data, predictions, prediction_info)
        
        # Generate output path if not provided
        if not args.output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = os.path.basename(args.model_path)
            args.output_path = f'predictions_{model_name}_{timestamp}.{args.output_format}'
        
        # Save predictions
        logging.info(f"Saving predictions in {args.output_format} format...")
        save_predictions(results, args.output_path, args.output_format)
        
        # Log prediction summary
        logging.info("\nPrediction Summary:")
        logging.info(f"Total predictions: {len(predictions)}")
        logging.info(f"Prediction range: {results.index[0]} to {results.index[-1]}")
        logging.info(f"Output saved to: {args.output_path}")
        logging.info("Prediction process completed successfully")
        
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main() 