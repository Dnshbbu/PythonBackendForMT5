import sqlite3
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_data():
    # Connect to the database
    conn = sqlite3.connect('logs/trading_data.db')
    
    # Get schema
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='historical_predictions'")
    schema = cursor.fetchone()[0]
    logging.info(f"\nTable schema:\n{schema}")
    
    # Get data for specific run_ids
    run_ids = ['run_20250207_234606', 'run_20250207_204318']
    query = """
    SELECT 
        datetime,
        actual_price,
        predicted_price,
        error,
        price_change,
        predicted_change,
        price_volatility,
        model_name,
        run_id
    FROM historical_predictions
    WHERE run_id IN (?, ?)
    ORDER BY datetime
    LIMIT 5
    """
    
    df = pd.read_sql_query(query, conn, params=run_ids)
    logging.info(f"\nSample data for run_ids {run_ids}:\n{df}")
    
    # Get unique model names for these run_ids
    query = """
    SELECT DISTINCT model_name
    FROM historical_predictions
    WHERE run_id IN (?, ?)
    """
    
    df_models = pd.read_sql_query(query, conn, params=run_ids)
    logging.info(f"\nUnique model names for these run_ids:\n{df_models}")
    
    conn.close()

if __name__ == "__main__":
    check_data() 