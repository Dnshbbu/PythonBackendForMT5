import sqlite3
import pandas as pd
from typing import List

def get_table_names(db_path: str) -> List[str]:
    """Get list of table names from the database"""
    try:
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
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 1", conn)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        return numeric_cols
    finally:
        if conn:
            conn.close() 