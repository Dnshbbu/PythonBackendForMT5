import argparse
import logging
import os
import pandas as pd
import json
from datetime import datetime
from time_series_predictor import TimeSeriesPredictor
from typing import List, Dict

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
    
    return parser.parse_args()

def format_predictions(data: pd.DataFrame, predictions: List[float], prediction_info: Dict) -> pd.DataFrame:
    """Format predictions into a DataFrame
    
    Args:
        data: Original data
        predictions: Model predictions
        prediction_info: Additional prediction information
    """
    results = pd.DataFrame()
    results['DateTime'] = data.index
    results['Actual'] = data['Price']
    
    # Handle single prediction (for the last point)
    if len(predictions) == 1:
        results['Predicted'] = None
        results.iloc[-1, results.columns.get_loc('Predicted')] = predictions[0]
    else:
        results['Predicted'] = predictions
    
    # Handle confidence intervals
    if prediction_info.get('lower_bound') is not None:
        if len(prediction_info['lower_bound']) == 1:
            results['Lower_Bound'] = None
            results.iloc[-1, results.columns.get_loc('Lower_Bound')] = prediction_info['lower_bound'][0]
        else:
            results['Lower_Bound'] = prediction_info['lower_bound']
    
    if prediction_info.get('upper_bound') is not None:
        if len(prediction_info['upper_bound']) == 1:
            results['Upper_Bound'] = None
            results.iloc[-1, results.columns.get_loc('Upper_Bound')] = prediction_info['upper_bound'][0]
        else:
            results['Upper_Bound'] = prediction_info['upper_bound']
    
    return results

def main():
    """Main function to run predictions"""
    args = parse_args()
    setup_logging()
    
    try:
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        
        if not args.output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = os.path.basename(args.model_path)
            args.output_path = f"predictions_{model_name}_{timestamp}.{args.output_format}"
        
        # Load model
        predictor = TimeSeriesPredictor(args.model_path)
        
        # Show model info if requested
        if args.show_metrics:
            model_info = predictor.get_model_info()
            print("\nModel Information:")
            print("-" * 50)
            for key, value in model_info.items():
                print(f"{key}: {value}")
            print("-" * 50)
        
        # Load data
        logging.info(f"Loading data from table {args.table}")
        data = load_data_from_db(
            db_path,
            args.table,
            args.start_date,
            args.end_date
        )
        
        # Validate data
        if not predictor.validate_data(data):
            raise ValueError("Data validation failed")
        
        # Make predictions
        logging.info("Making predictions...")
        predictions, prediction_info = predictor.predict(data)
        
        # Format and save predictions
        results = format_predictions(data, predictions, prediction_info)
        save_predictions(results, args.output_path, args.output_format)
        
        logging.info("Prediction completed successfully")
        
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 