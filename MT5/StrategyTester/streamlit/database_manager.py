import sqlite3
import os
import pandas as pd
import logging
from typing import Dict, List, Any

class DatabaseManager:
    def __init__(self, db_path: str = 'trading_data.db'):
        """Initialize DatabaseManager with database path"""
        self.db_path = db_path
        self.table_columns = {}  # Cache for table columns
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def create_connection(self):
        """Create a database connection"""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
            logging.error(f"Error connecting to database: {e}")
            raise

    def get_max_splits(self, data: Dict[str, Any]) -> Dict[str, int]:
        """Determine the maximum number of splits needed for each column"""
        split_counts = {}
        split_columns = ['Factors', 'ExitFactors', 'EntryScore', 'ExitScoreDetails', 'Pullback']
        
        for column in split_columns:
            if column in data and data[column]:
                splits = data[column].split('|')
                split_counts[column] = len(splits)
            else:
                split_counts[column] = 0
                
        return split_counts
            
    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data by splitting specified columns by '|' delimiter"""
        processed_data = data.copy()
        
        # Columns to split
        split_columns = ['Factors', 'ExitFactors', 'EntryScore', 'ExitScoreDetails', 'Pullback']
        
        for column in split_columns:
            if column in processed_data and processed_data[column]:
                # Split the column value by '|' and create new columns
                split_values = processed_data[column].split('|')
                # Remove the original column
                processed_data.pop(column)
                
                # Add new columns with prefix
                for i, value in enumerate(split_values, 1):
                    if '=' in value:
                        # Handle key-value pairs
                        key, val = value.split('=', 1)
                        new_column = f"{column}_{key.strip()}"
                        processed_data[new_column] = val.strip()
                    else:
                        # Handle simple values
                        new_column = f"{column}_{i}"
                        processed_data[new_column] = value.strip()
                    
        return processed_data

    def get_all_column_names(self, data: Dict[str, Any]) -> List[str]:
        """Get all column names after preprocessing"""
        processed_data = self.preprocess_data(data)
        return list(processed_data.keys())
        
    def create_table_for_strategy(self, run_id: str, sample_data: Dict[str, Any]):
        """Create table for a specific strategy if it doesn't exist"""
        try:
            conn = self.create_connection()
            cursor = conn.cursor()
            
            # Process sample data to get all possible columns
            processed_data = self.preprocess_data(sample_data)
            
            # Base columns with their types
            columns = []
            for col_name in processed_data.keys():
                # Determine column type based on value
                value = processed_data[col_name]
                if isinstance(value, (int, bool)):
                    col_type = 'INTEGER'
                elif isinstance(value, float):
                    col_type = 'REAL'
                else:
                    col_type = 'TEXT'
                columns.append(f'{col_name} {col_type}')
            
            # Create table query
            table_name = f"strategy_{run_id}"
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {', '.join(columns)}
            )
            """
            
            cursor.execute(create_table_query)
            conn.commit()
            
            # Cache the column names for this table
            self.table_columns[table_name] = set(processed_data.keys())
            
            logging.info(f"Created/verified table: {table_name}")
            return table_name
            
        except sqlite3.Error as e:
            logging.error(f"Error creating table: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def alter_table_add_columns(self, table_name: str, new_columns: List[str]):
        """Add new columns to existing table"""
        try:
            conn = self.create_connection()
            cursor = conn.cursor()
            
            for column in new_columns:
                try:
                    alter_query = f"ALTER TABLE {table_name} ADD COLUMN {column} TEXT"
                    cursor.execute(alter_query)
                    self.table_columns[table_name].add(column)
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e):
                        raise
            
            conn.commit()
            
        except sqlite3.Error as e:
            logging.error(f"Error altering table: {e}")
            raise
        finally:
            if conn:
                conn.close()
                
    def insert_data(self, table_name: str, data: Dict[str, Any]):
        """Insert preprocessed data into the specified table"""
        try:
            # Process the data
            processed_data = self.preprocess_data(data)
            
            # Check if we need to add new columns
            if table_name in self.table_columns:
                new_columns = set(processed_data.keys()) - self.table_columns[table_name]
                if new_columns:
                    self.alter_table_add_columns(table_name, list(new_columns))
            
            conn = self.create_connection()
            cursor = conn.cursor()
            
            # Get the columns that exist in the data
            columns = list(processed_data.keys())
            placeholders = ','.join(['?' for _ in columns])
            
            # Create the insert query
            insert_query = f"""
            INSERT INTO {table_name} 
            ({','.join(columns)}) 
            VALUES ({placeholders})
            """
            
            # Execute the insert
            cursor.execute(insert_query, list(processed_data.values()))
            conn.commit()
            logging.info(f"Inserted new row into {table_name}")
            
        except sqlite3.Error as e:
            logging.error(f"Error inserting data: {e}")
            raise
        finally:
            if conn:
                conn.close()