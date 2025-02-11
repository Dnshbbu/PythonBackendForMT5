import sqlite3
import pandas as pd
from datetime import datetime
from typing import List
import os

def get_table_names(db_path: str) -> List[str]:
    """Get list of table names from the database"""
    try:
        # Update the db_path to point to the logs directory if needed
        if not os.path.exists(db_path) and os.path.exists(os.path.join('logs', db_path)):
            db_path = os.path.join('logs', db_path)
            
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor]
        return tables
    finally:
        if conn:
            conn.close()

def get_numeric_columns(db_path: str, table_name: str) -> List[str]:
    """Get list of numeric column names from a table"""
    try:
        # Update the db_path to point to the logs directory if needed
        if not os.path.exists(db_path) and os.path.exists(os.path.join('logs', db_path)):
            db_path = os.path.join('logs', db_path)
            
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 1", conn)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        return numeric_cols
    finally:
        if conn:
            conn.close()

def get_table_info():
    conn = sqlite3.connect('logs/trading_data.db')
    cursor = conn.cursor()
    
    # Get list of strategy tables
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name LIKE 'strategy_%'
        ORDER BY name DESC
    """)
    
    tables = cursor.fetchall()
    
    print("\nTable Information:")
    print("-" * 80)
    
    for table in tables:
        table_name = table[0]
        print(f"\nTable: {table_name}")
        
        # Get schema first
        cursor.execute(f'PRAGMA table_info({table_name})')
        schema = cursor.fetchall()
        print("\nColumns:")
        for col in schema:
            print(f"  {col[1]}: {col[2]}")
        
        # Get first and last row timestamps (using time column)
        cursor.execute(f'SELECT time FROM {table_name} ORDER BY time ASC LIMIT 1')
        first_ts = cursor.fetchone()[0]
        cursor.execute(f'SELECT time FROM {table_name} ORDER BY time DESC LIMIT 1')
        last_ts = cursor.fetchone()[0]
        
        # Get row count
        cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
        count = cursor.fetchone()[0]
        
        # Get unique symbols
        cursor.execute(f'SELECT DISTINCT symbol FROM {table_name}')
        symbols = [row[0] for row in cursor.fetchall()]
        
        print(f"\nSummary:")
        print(f"  Date Range: {first_ts} to {last_ts}")
        print(f"  Total Rows: {count}")
        print(f"  Symbols: {', '.join(symbols)}")
    
    conn.close()

if __name__ == "__main__":
    get_table_info() 