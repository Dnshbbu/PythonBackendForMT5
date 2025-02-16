import sqlite3
import logging
import os

# Constants
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'trading_data.db')

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def migrate_historical_predictions():
    """Add forecast_horizon column to historical_predictions table"""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        
        # Check if column exists
        cursor.execute("PRAGMA table_info(historical_predictions)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'forecast_horizon' not in columns:
            # Create temporary table with new schema
            cursor.execute('''
                CREATE TABLE historical_predictions_new (
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
                    model_name TEXT,
                    forecast_horizon INTEGER DEFAULT 1
                )
            ''')
            
            # Copy data from old table to new table
            cursor.execute('''
                INSERT INTO historical_predictions_new 
                (datetime, actual_price, predicted_price, error, price_change, predicted_change, 
                price_volatility, run_id, source_table, model_name)
                SELECT datetime, actual_price, predicted_price, error, price_change, predicted_change, 
                price_volatility, run_id, source_table, model_name
                FROM historical_predictions
            ''')
            
            # Drop old table
            cursor.execute('DROP TABLE historical_predictions')
            
            # Rename new table to original name
            cursor.execute('ALTER TABLE historical_predictions_new RENAME TO historical_predictions')
            
            conn.commit()
            logging.info("Successfully added forecast_horizon column to historical_predictions table")
        else:
            logging.info("forecast_horizon column already exists in historical_predictions table")
            
    except Exception as e:
        logging.error(f"Error during migration: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
        raise
    finally:
        if 'conn' in locals():
            conn.close()

def migrate_historical_prediction_metrics():
    """Add forecast_horizon column to historical_prediction_metrics table"""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        
        # Check if column exists
        cursor.execute("PRAGMA table_info(historical_prediction_metrics)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'forecast_horizon' not in columns:
            # Create temporary table with new schema
            cursor.execute('''
                CREATE TABLE historical_prediction_metrics_new (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    run_id TEXT,
                    source_table TEXT,
                    model_name TEXT,
                    forecast_horizon INTEGER DEFAULT 1,
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
            ''')
            
            # Copy data from old table to new table
            cursor.execute('''
                INSERT INTO historical_prediction_metrics_new 
                (timestamp, run_id, source_table, model_name,
                total_predictions, mean_absolute_error, root_mean_squared_error,
                mean_absolute_percentage_error, r_squared, direction_accuracy,
                up_prediction_accuracy, down_prediction_accuracy, correct_ups,
                correct_downs, total_ups, total_downs, max_error,
                min_error, std_error, avg_price_change, price_volatility,
                mean_prediction_error, median_prediction_error, error_skewness,
                first_quarter_accuracy, last_quarter_accuracy,
                max_correct_streak, avg_correct_streak)
                SELECT timestamp, run_id, source_table, model_name,
                total_predictions, mean_absolute_error, root_mean_squared_error,
                mean_absolute_percentage_error, r_squared, direction_accuracy,
                up_prediction_accuracy, down_prediction_accuracy, correct_ups,
                correct_downs, total_ups, total_downs, max_error,
                min_error, std_error, avg_price_change, price_volatility,
                mean_prediction_error, median_prediction_error, error_skewness,
                first_quarter_accuracy, last_quarter_accuracy,
                max_correct_streak, avg_correct_streak
                FROM historical_prediction_metrics
            ''')
            
            # Drop old table
            cursor.execute('DROP TABLE historical_prediction_metrics')
            
            # Rename new table to original name
            cursor.execute('ALTER TABLE historical_prediction_metrics_new RENAME TO historical_prediction_metrics')
            
            conn.commit()
            logging.info("Successfully added forecast_horizon column to historical_prediction_metrics table")
        else:
            logging.info("forecast_horizon column already exists in historical_prediction_metrics table")
            
    except Exception as e:
        logging.error(f"Error during migration: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
        raise
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    """Main function to run all migrations"""
    setup_logging()
    logging.info("Starting database migrations...")
    
    try:
        migrate_historical_predictions()
        migrate_historical_prediction_metrics()
        logging.info("All migrations completed successfully")
    except Exception as e:
        logging.error(f"Migration failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 