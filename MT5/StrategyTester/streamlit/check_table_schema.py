import sqlite3
import pandas as pd
import os

def check_table_info(table_name: str):
    """Check table schema and sample data"""
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        
        # Get table schema
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        schema = cursor.fetchall()
        
        print("\nTable Schema:")
        print("-" * 80)
        for col in schema:
            print(f"Column: {col[1]:<20} Type: {col[2]:<15} Nullable: {col[3]}")
            
        # Get sample data
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
        
        print("\nSample Data:")
        print("-" * 80)
        print(df)
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print("\nNumeric Columns:")
        print("-" * 80)
        print(numeric_cols)
        
        return numeric_cols
        
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    table_name = "strategy_TRIP_NAS_10032544"
    numeric_cols = check_table_info(table_name) 