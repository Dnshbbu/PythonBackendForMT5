import argparse
import os
import pandas as pd
import sqlite3
import logging
from pycaret.regression import load_model, predict_model
from datetime import datetime

def setup_logging():
    """Setup logging configuration"""
    log_file = f'pycaret_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_data_from_db(db_path: str, table_name: str) -> pd.DataFrame:
    """Load data from SQLite database"""
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        logging.info(f"Loaded {len(df)} rows from table {table_name}")
        return df
    finally:
        if conn:
            conn.close()

def prepare_features(df: pd.DataFrame, n_lags: int = 3) -> pd.DataFrame:
    """Prepare features including lag features if used during training"""
    # Create lag features for Price column if it exists
    if 'Price' in df.columns:
        for i in range(1, n_lags + 1):
            df[f'Price_lag_{i}'] = df['Price'].shift(i)
    
    # Drop rows with NaN values created by lag features
    df = df.dropna()
    logging.info(f"Prepared features with {n_lags} lags. Shape after preparation: {df.shape}")
    return df

def save_predictions(predictions: pd.DataFrame, output_file: str):
    """Save predictions to CSV file"""
    predictions.to_csv(output_file, index=False)
    logging.info(f"Saved predictions to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained PyCaret model')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Name of the trained model (e.g., pycaret-XGBoost_single_20250213_223546)')
    parser.add_argument('--table_name', type=str, required=True,
                       help='Name of the table to make predictions on')
    parser.add_argument('--n_lags', type=int, default=3,
                       help='Number of lag features used during training (default: 3)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file path for predictions (default: predictions_[model_name]_[timestamp].csv)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Set paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        models_dir = os.path.join(current_dir, 'models')
        # Remove .pkl extension since PyCaret adds it automatically
        model_path = os.path.join(models_dir, args.model_name, "model")
        
        if not os.path.exists(model_path + ".pkl"):
            raise FileNotFoundError(f"Model file not found: {model_path}.pkl")
        
        # Set default output file if not provided
        if args.output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_file = f"predictions_{args.model_name}_{timestamp}.csv"
        
        # Load data
        logger.info("Loading data from database...")
        df = load_data_from_db(db_path, args.table_name)
        
        # Prepare features
        logger.info("Preparing features...")
        df_prepared = prepare_features(df, args.n_lags)
        
        # Load model
        logger.info(f"Loading model from {model_path}...")
        model = load_model(model_path)
        
        # Make predictions using the model's pipeline directly
        logger.info("Making predictions...")
        try:
            # First try using the model's pipeline directly
            predictions = model.predict(df_prepared)
            predictions_df = pd.DataFrame({'Prediction': predictions})
            # Add original data columns
            for col in df_prepared.columns:
                predictions_df[col] = df_prepared[col]
        except Exception as e:
            logger.warning(f"Direct prediction failed, trying with predict_model: {str(e)}")
            predictions_df = predict_model(model, data=df_prepared)
        
        # Save predictions
        save_predictions(predictions_df, args.output_file)
        
        logger.info("Prediction process completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 