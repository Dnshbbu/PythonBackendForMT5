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




    def get_all_column_names(self, data: Dict[str, Any]) -> List[str]:
        """Get all column names after preprocessing"""
        processed_data = self.preprocess_data(data)
        return list(processed_data.keys())
        

                

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





    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data by splitting specified columns and handling nested structures"""
        processed_data = data.copy()
        
        # Columns to split
        split_columns = ['Factors', 'ExitFactors', 'EntryScore', 'ExitScoreDetails', 'Pullback']
        
        for column in split_columns:
            if column in processed_data and processed_data[column]:
                # Split the column value by '|' and create new columns
                split_values = processed_data[column].split('|')
                processed_data.pop(column)  # Remove the original column
                
                # Process each split value
                for value in split_values:
                    if '=' in value:
                        # Handle key-value pairs
                        key, val = value.split('=', 1)
                        new_column = f"{column}_{key.strip()}"
                    else:
                        # Handle complex nested structures (like fvgContext)
                        parts = value.split(':')
                        if len(parts) > 1:
                            key = parts[0].strip()
                            val = ':'.join(parts[1:]).strip()
                            new_column = f"{column}_{key}"
                        else:
                            # Handle simple values or empty strings
                            continue
                    
                    # Convert numeric values
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        pass
                    
                    processed_data[new_column] = val
        
        return processed_data

    def create_table_for_strategy(self, run_id: str, sample_data: Dict[str, Any]) -> str:
        """Create table for a specific strategy with dynamic column handling"""
        try:
            conn = self.create_connection()
            cursor = conn.cursor()
            
            # Process sample data to get all possible columns
            processed_data = self.preprocess_data(sample_data)
            
            # Table name
            table_name = f"strategy_{run_id}"
            
            # Check if table exists
            cursor.execute(f"""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (table_name,))
            
            table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                # Create initial columns
                columns = []
                for col_name, value in processed_data.items():
                    col_type = 'REAL' if isinstance(value, (int, float)) else 'TEXT'
                    columns.append(f'{col_name} {col_type}')
                
                create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    {', '.join(columns)}
                )
                """
                cursor.execute(create_table_query)
                
            # Cache the column names
            cursor.execute(f"PRAGMA table_info({table_name})")
            self.table_columns[table_name] = {row[1] for row in cursor.fetchall()}
            
            conn.commit()
            logging.info(f"Created/verified table: {table_name}")
            return table_name
            
        except sqlite3.Error as e:
            logging.error(f"Error creating table: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def alter_table_add_columns(self, table_name: str, new_columns: List[str]):
        """Add new columns to existing table with better error handling"""
        try:
            conn = self.create_connection()
            cursor = conn.cursor()
            
            for column in new_columns:
                try:
                    # Check if column exists
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    existing_columns = {row[1] for row in cursor.fetchall()}
                    
                    if column not in existing_columns:
                        alter_query = f"ALTER TABLE {table_name} ADD COLUMN {column} TEXT"
                        cursor.execute(alter_query)
                        self.table_columns[table_name].add(column)
                        logging.info(f"Added new column: {column} to table: {table_name}")
                
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e):
                        logging.error(f"Error adding column {column}: {e}")
                        raise
            
            conn.commit()
            
        except sqlite3.Error as e:
            logging.error(f"Error altering table: {e}")
            raise
        finally:
            if conn:
                conn.close()

        


