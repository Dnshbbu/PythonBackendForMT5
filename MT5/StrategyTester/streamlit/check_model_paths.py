import sqlite3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connect to the database
db_path = r"C:\Users\StdUser\Desktop\MyProjects\Backtesting\MT5\StrategyTester\streamlit\logs\trading_data.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Query the models table for recent entries
    cursor.execute("""
        SELECT model_name, model_type, model_path, scaler_path 
        FROM models 
        WHERE model_name LIKE 'ts_multi_test1%'
        ORDER BY created_at DESC
    """)
    
    models = cursor.fetchall()
    
    logging.info(f"Found {len(models)} models:")
    for model in models:
        model_name, model_type, model_path, scaler_path = model
        logging.info(f"\nModel Name: {model_name}")
        logging.info(f"Model Type: {model_type}")
        logging.info(f"Model Path: {model_path}")
        logging.info(f"Scaler Path: {scaler_path}")
        
except sqlite3.Error as e:
    logging.error(f"Database error: {e}")
finally:
    conn.close() 