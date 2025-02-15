import sqlite3
import argparse
import pandas as pd
from typing import Optional, Union, List
import logging

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_sample_data(
    db_path: str,
    table_name: str,
    columns: Optional[List[str]] = None,
    limit: int = 5,
    order_by: Optional[str] = None,
    where_clause: Optional[str] = None,
    random_sample: bool = False
) -> pd.DataFrame:
    """
    Get sample data from a specific table in SQLite database.
    
    Args:
        db_path: Path to the SQLite database
        table_name: Name of the table to get data from
        columns: List of columns to retrieve (None for all columns)
        limit: Number of rows to retrieve
        order_by: Column to order by (with optional ASC/DESC)
        where_clause: Optional WHERE clause for filtering
        random_sample: If True, returns random samples instead of first N rows
        
    Returns:
        DataFrame containing the sample data
    """
    try:
        conn = sqlite3.connect(db_path)
        
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name 
            FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (table_name,))
        
        if not cursor.fetchone():
            raise ValueError(f"Table '{table_name}' does not exist in the database")
        
        # Build query
        cols = ', '.join(columns) if columns else '*'
        query = f"SELECT {cols} FROM {table_name}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
            
        if random_sample:
            query += " ORDER BY RANDOM()"
        elif order_by:
            query += f" ORDER BY {order_by}"
            
        query += f" LIMIT {limit}"
        
        # Get data
        df = pd.read_sql_query(query, conn)
        
        # Get total row count for context
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        
        logging.info(f"Retrieved {len(df)} samples out of {total_rows} total rows")
        return df
        
    except Exception as e:
        raise Exception(f"Error getting sample data: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

def print_sample_data(df: pd.DataFrame):
    """Print sample data in a readable format"""
    if df.empty:
        print("\nNo data found.")
        return
        
    print("\nSample Data:")
    print("-" * 100)
    
    # Print column names with types
    col_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        col_info.append(f"{col} ({dtype})")
    print("Columns:", ", ".join(col_info))
    print("-" * 100)
    
    # Print the data
    print(df.to_string(index=True))
    print("-" * 100)
    
    # Print basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if not numeric_cols.empty:
        print("\nNumeric Column Statistics:")
        print("-" * 100)
        stats = df[numeric_cols].describe()
        print(stats)
        print("-" * 100)

def main():
    parser = argparse.ArgumentParser(description='Get sample data from a SQLite database table')
    
    # Required arguments
    parser.add_argument('db_path', help='Path to the SQLite database')
    parser.add_argument('table_name', help='Name of the table to get data from')
    
    # Optional arguments
    parser.add_argument('--columns', help='Comma-separated list of columns to retrieve')
    parser.add_argument('--limit', type=int, default=5, help='Number of rows to retrieve (default: 5)')
    parser.add_argument('--order-by', help='Column to order by (with optional ASC/DESC)')
    parser.add_argument('--where', help='WHERE clause for filtering data')
    parser.add_argument('--random', action='store_true', help='Get random samples instead of first N rows')
    
    args = parser.parse_args()
    setup_logging()
    
    try:
        # Convert columns string to list if provided
        columns = args.columns.split(',') if args.columns else None
        
        # Get and print the data
        df = get_sample_data(
            args.db_path,
            args.table_name,
            columns=columns,
            limit=args.limit,
            order_by=args.order_by,
            where_clause=args.where,
            random_sample=args.random
        )
        
        print_sample_data(df)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 