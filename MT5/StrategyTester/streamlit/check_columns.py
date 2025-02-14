import sqlite3
import pandas as pd
import os

def get_table_info(table_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
    
    conn = sqlite3.connect(db_path)
    try:
        # Get column information
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Get a sample to identify numeric columns
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 1", conn)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print("\nTable Information:")
        print("-" * 50)
        print("\nAll columns:")
        for i, col in enumerate(columns, 1):
            print(f"{i}. {col}")
        
        print("\nNumeric columns:")
        for i, col in enumerate(numeric_cols, 1):
            print(f"{i}. {col}")
        
    finally:
        conn.close()

if __name__ == "__main__":
    table_name = 'strategy_TRIP_NAS_10031622'
    get_table_info(table_name) 