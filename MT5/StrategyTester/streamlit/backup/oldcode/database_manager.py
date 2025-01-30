# import sqlite3
# import os
# import pandas as pd
# import logging
# from typing import Dict, List, Any

# class DatabaseManager:
#     def __init__(self, db_path: str = 'trading_data.db'):
#         """Initialize DatabaseManager with database path"""
#         self.db_path = db_path
#         self.table_columns = {}  # Cache for table columns
#         self.setup_logging()
        
#     def setup_logging(self):
#         """Setup logging configuration"""
#         logging.basicConfig(
#             level=logging.INFO,
#             format='%(asctime)s - %(levelname)s - %(message)s'
#         )
        
#     def create_connection(self):
#         """Create a database connection"""
#         try:
#             conn = sqlite3.connect(self.db_path)
#             return conn
#         except sqlite3.Error as e:
#             logging.error(f"Error connecting to database: {e}")
#             raise

#     def get_max_splits(self, data: Dict[str, Any]) -> Dict[str, int]:
#         """Determine the maximum number of splits needed for each column"""
#         split_counts = {}
#         split_columns = ['Factors', 'ExitFactors', 'EntryScore', 'ExitScoreDetails', 'Pullback']
        
#         for column in split_columns:
#             if column in data and data[column]:
#                 splits = data[column].split('|')
#                 split_counts[column] = len(splits)
#             else:
#                 split_counts[column] = 0
                
#         return split_counts




#     def get_all_column_names(self, data: Dict[str, Any]) -> List[str]:
#         """Get all column names after preprocessing"""
#         processed_data = self.preprocess_data(data)
#         return list(processed_data.keys())
        

                

#     def insert_data(self, table_name: str, data: Dict[str, Any]):
#         """Insert preprocessed data into the specified table"""
#         try:
#             # Process the data
#             processed_data = self.preprocess_data(data)
            
#             # Check if we need to add new columns
#             if table_name in self.table_columns:
#                 new_columns = set(processed_data.keys()) - self.table_columns[table_name]
#                 if new_columns:
#                     self.alter_table_add_columns(table_name, list(new_columns))
            
#             conn = self.create_connection()
#             cursor = conn.cursor()
            
#             # Get the columns that exist in the data
#             columns = list(processed_data.keys())
#             placeholders = ','.join(['?' for _ in columns])
            
#             # Create the insert query
#             insert_query = f"""
#             INSERT INTO {table_name} 
#             ({','.join(columns)}) 
#             VALUES ({placeholders})
#             """
            
#             # Execute the insert
#             cursor.execute(insert_query, list(processed_data.values()))
#             conn.commit()
#             logging.info(f"Inserted new row into {table_name}")
            
#         except sqlite3.Error as e:
#             logging.error(f"Error inserting data: {e}")
#             raise
#         finally:
#             if conn:
#                 conn.close()





#     def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
#         """Preprocess data by splitting specified columns and handling nested structures"""
#         processed_data = data.copy()
        
#         # Columns to split
#         split_columns = ['Factors', 'ExitFactors', 'EntryScore', 'ExitScoreDetails', 'Pullback']
        
#         for column in split_columns:
#             if column in processed_data and processed_data[column]:
#                 # Split the column value by '|' and create new columns
#                 split_values = processed_data[column].split('|')
#                 processed_data.pop(column)  # Remove the original column
                
#                 # Process each split value
#                 for value in split_values:
#                     if '=' in value:
#                         # Handle key-value pairs
#                         key, val = value.split('=', 1)
#                         new_column = f"{column}_{key.strip()}"
#                     else:
#                         # Handle complex nested structures (like fvgContext)
#                         parts = value.split(':')
#                         if len(parts) > 1:
#                             key = parts[0].strip()
#                             val = ':'.join(parts[1:]).strip()
#                             new_column = f"{column}_{key}"
#                         else:
#                             # Handle simple values or empty strings
#                             continue
                    
#                     # Convert numeric values
#                     try:
#                         val = float(val)
#                     except (ValueError, TypeError):
#                         pass
                    
#                     processed_data[new_column] = val
        
#         return processed_data

#     def create_table_for_strategy(self, run_id: str, sample_data: Dict[str, Any]) -> str:
#         """Create table for a specific strategy with dynamic column handling"""
#         try:
#             conn = self.create_connection()
#             cursor = conn.cursor()
            
#             # Process sample data to get all possible columns
#             processed_data = self.preprocess_data(sample_data)
            
#             # Table name
#             table_name = f"strategy_{run_id}"
            
#             # Check if table exists
#             cursor.execute(f"""
#                 SELECT name FROM sqlite_master 
#                 WHERE type='table' AND name=?
#             """, (table_name,))
            
#             table_exists = cursor.fetchone() is not None
            
#             if not table_exists:
#                 # Create initial columns
#                 columns = []
#                 for col_name, value in processed_data.items():
#                     col_type = 'REAL' if isinstance(value, (int, float)) else 'TEXT'
#                     columns.append(f'{col_name} {col_type}')
                
#                 create_table_query = f"""
#                 CREATE TABLE IF NOT EXISTS {table_name} (
#                     id INTEGER PRIMARY KEY AUTOINCREMENT,
#                     {', '.join(columns)}
#                 )
#                 """
#                 cursor.execute(create_table_query)
                
#             # Cache the column names
#             cursor.execute(f"PRAGMA table_info({table_name})")
#             self.table_columns[table_name] = {row[1] for row in cursor.fetchall()}
            
#             conn.commit()
#             logging.info(f"Created/verified table: {table_name}")
#             return table_name
            
#         except sqlite3.Error as e:
#             logging.error(f"Error creating table: {e}")
#             raise
#         finally:
#             if conn:
#                 conn.close()

#     def alter_table_add_columns(self, table_name: str, new_columns: List[str]):
#         """Add new columns to existing table with better error handling"""
#         try:
#             conn = self.create_connection()
#             cursor = conn.cursor()
            
#             for column in new_columns:
#                 try:
#                     # Check if column exists
#                     cursor.execute(f"PRAGMA table_info({table_name})")
#                     existing_columns = {row[1] for row in cursor.fetchall()}
                    
#                     if column not in existing_columns:
#                         alter_query = f"ALTER TABLE {table_name} ADD COLUMN {column} TEXT"
#                         cursor.execute(alter_query)
#                         self.table_columns[table_name].add(column)
#                         logging.info(f"Added new column: {column} to table: {table_name}")
                
#                 except sqlite3.OperationalError as e:
#                     if "duplicate column name" not in str(e):
#                         logging.error(f"Error adding column {column}: {e}")
#                         raise
            
#             conn.commit()
            
#         except sqlite3.Error as e:
#             logging.error(f"Error altering table: {e}")
#             raise
#         finally:
#             if conn:
#                 conn.close()

        




import sqlite3
import os
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from config import DATABASE_CONFIG

class DatabaseManager:
    def __init__(self):
        """Initialize DatabaseManager with database path from config"""
        self.db_path = os.path.join(DATABASE_CONFIG['logs_dir'], DATABASE_CONFIG['db_name'])
        self.table_columns = {}  # Cache for table columns
        self.setup_logging()
        self.initialize_database()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def initialize_database(self):
        """Ensure database and required directories exist"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Test connection and create database if it doesn't exist
            conn = self.create_connection()
            if conn:
                conn.close()
                logging.info(f"Database initialized successfully at {self.db_path}")
        except Exception as e:
            logging.error(f"Error initializing database: {e}")
            raise

    def create_connection(self):
        """Create a database connection with error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
            logging.error(f"Error connecting to database at {self.db_path}: {e}")
            raise

    def get_max_splits(self, data: Dict[str, Any]) -> Dict[str, int]:
        """
        Determine the maximum number of splits needed for each column
        
        Args:
            data: Dictionary containing the data to analyze
            
        Returns:
            Dictionary with column names and their maximum split counts
        """
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
        """
        Get all column names after preprocessing
        
        Args:
            data: Dictionary containing the data to analyze
            
        Returns:
            List of all column names
        """
        processed_data = self.preprocess_data(data)
        return list(processed_data.keys())

    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess data by splitting specified columns and handling nested structures
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Processed data dictionary
        """
        try:
            processed_data = {}
            
            if not data:
                logging.warning("Empty data received for preprocessing")
                return processed_data
                
            # Copy and convert basic fields
            for key, value in data.items():
                if value is None:
                    processed_data[key] = ''
                    continue
                    
                try:
                    if isinstance(value, str) and value.strip():
                        if '.' in value:
                            processed_data[key] = float(value)
                        else:
                            processed_data[key] = int(value)
                    else:
                        processed_data[key] = value
                except ValueError:
                    processed_data[key] = value
            
            # Process special columns
            split_columns = [
                'Factors', 
                'ExitFactors', 
                'EntryScore', 
                'ExitScoreDetails', 
                'Pullback'
            ]
            
            for column in split_columns:
                if column in processed_data and processed_data[column]:
                    column_value = processed_data[column]
                    processed_data.pop(column)
                    
                    if not isinstance(column_value, str):
                        continue
                    
                    split_values = column_value.split('|')
                    for value in split_values:
                        if not value.strip():
                            continue
                            
                        if '=' in value:
                            key, val = value.split('=', 1)
                            new_column = f"{column}_{key.strip()}"
                        else:
                            parts = value.split(':')
                            if len(parts) > 1:
                                key = parts[0].strip()
                                val = ':'.join(parts[1:]).strip()
                                new_column = f"{column}_{key}"
                            else:
                                continue
                        
                        # Convert values
                        try:
                            if val.strip().lower() == 'true':
                                processed_val = 1
                            elif val.strip().lower() == 'false':
                                processed_val = 0
                            elif '.' in val:
                                processed_val = float(val)
                            else:
                                processed_val = int(val)
                        except (ValueError, TypeError):
                            processed_val = val
                        
                        processed_data[new_column] = processed_val
            
            return processed_data
            
        except Exception as e:
            logging.error(f"Error preprocessing data: {e}")
            raise

    def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate the data structure and required fields
        
        Args:
            data: Data dictionary to validate
            
        Returns:
            True if validation passes
        """
        try:
            if not data:
                raise ValueError("Empty data received")
            
            required_fields = ['Date', 'Time', 'Symbol', 'Price']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            numeric_fields = ['Price', 'Score', 'ExitScore', 'Equity', 'Balance', 'Profit']
            for field in numeric_fields:
                if field in data and data[field]:
                    try:
                        float(data[field])
                    except ValueError:
                        raise ValueError(f"Field {field} contains non-numeric value: {data[field]}")
            
            return True
            
        except Exception as e:
            logging.error(f"Data validation failed: {e}")
            return False

    def get_sql_type(self, value: Any) -> str:
        """
        Determine appropriate SQL type for a given Python value
        
        Args:
            value: Python value to convert
            
        Returns:
            SQL type name
        """
        if isinstance(value, bool):
            return 'INTEGER'
        elif isinstance(value, int):
            return 'INTEGER'
        elif isinstance(value, float):
            return 'REAL'
        elif isinstance(value, (list, dict)):
            return 'TEXT'
        else:
            return 'TEXT'



    # def create_table_for_strategy(self, run_id: str, sample_data: Dict[str, Any]) -> str:
    #     """
    #     Create table for a specific strategy with dynamic column handling
        
    #     Args:
    #         run_id: Strategy run identifier
    #         sample_data: Sample data to determine schema
            
    #     Returns:
    #         Table name
    #     """
    #     conn = None
    #     try:
    #         conn = self.create_connection()
    #         cursor = conn.cursor()
            
    #         if not self.validate_data(sample_data):
    #             raise ValueError("Invalid data format")
            
    #         processed_data = self.preprocess_data(sample_data)
    #         table_name = f"strategy_{run_id}".replace('-', '_').replace(' ', '_')
            
    #         cursor.execute("""
    #             SELECT name FROM sqlite_master 
    #             WHERE type='table' AND name=?
    #         """, (table_name,))
            
    #         table_exists = cursor.fetchone() is not None
            
    #         if not table_exists:
    #             columns = []
    #             for col_name, value in processed_data.items():
    #                 col_type = self.get_sql_type(value)
    #                 columns.append(f'"{col_name}" {col_type}')
                
    #             create_table_query = f"""
    #             CREATE TABLE IF NOT EXISTS "{table_name}" (
    #                 id INTEGER PRIMARY KEY AUTOINCREMENT,
    #                 {', '.join(columns)}
    #             )
    #             """
    #             cursor.execute(create_table_query)
            
    #         cursor.execute(f'PRAGMA table_info("{table_name}")')
    #         self.table_columns[table_name] = {row[1] for row in cursor.fetchall()}
            
    #         conn.commit()
    #         logging.info(f"Created/verified table: {table_name}")
    #         return table_name
            
    #     except Exception as e:
    #         logging.error(f"Error creating table: {e}")
    #         raise
    #     finally:
    #         if conn:
    #             conn.close()


    def create_table_for_strategy(self, run_id: str, sample_data: Dict[str, Any]) -> str:
        """Create table for a specific strategy with dynamic column handling"""
        conn = None
        try:
            conn = self.create_connection()
            cursor = conn.cursor()
            
            # Process sample data to get all possible columns
            processed_data = self.preprocess_data(sample_data)
            
            # Sanitize table name
            table_name = self.sanitize_table_name(f"strategy_{run_id}")
            
            # Check if table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (table_name,))
            
            table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                # Create initial columns with proper SQL types
                columns = []
                for col_name, value in processed_data.items():
                    col_type = self.get_sql_type(value)
                    columns.append(f'"{col_name}" {col_type}')
                
                create_table_query = f"""
                CREATE TABLE IF NOT EXISTS "{table_name}" (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    {', '.join(columns)}
                )
                """
                cursor.execute(create_table_query)
                
            # Update column cache
            cursor.execute(f'PRAGMA table_info("{table_name}")')
            self.table_columns[table_name] = {row[1] for row in cursor.fetchall()}
            
            conn.commit()
            logging.info(f"Created/verified table: {table_name}")
            return table_name
            
        except Exception as e:
            logging.error(f"Error creating table: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def insert_data(self, table_name: str, data: Dict[str, Any]):
        """
        Insert preprocessed data into the specified table
        
        Args:
            table_name: Name of the target table
            data: Data to insert
        """
        conn = None
        try:
            processed_data = self.preprocess_data(data)
            
            if table_name in self.table_columns:
                new_columns = set(processed_data.keys()) - self.table_columns[table_name]
                if new_columns:
                    self.alter_table_add_columns(table_name, list(new_columns))
            
            conn = self.create_connection()
            cursor = conn.cursor()
            
            columns = list(processed_data.keys())
            placeholders = ','.join(['?' for _ in columns])
            
            insert_query = f"""
            INSERT INTO "{table_name}" 
            ({','.join(f'"{col}"' for col in columns)}) 
            VALUES ({placeholders})
            """
            
            cursor.execute(insert_query, list(processed_data.values()))
            conn.commit()
            logging.info(f"Inserted new row into {table_name}")
            
        except Exception as e:
            logging.error(f"Error inserting data: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def alter_table_add_columns(self, table_name: str, new_columns: List[str]):
        """
        Add new columns to existing table
        
        Args:
            table_name: Name of the table to alter
            new_columns: List of new columns to add
        """
        conn = None
        try:
            conn = self.create_connection()
            cursor = conn.cursor()
            
            for column in new_columns:
                try:
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    existing_columns = {row[1] for row in cursor.fetchall()}
                    
                    if column not in existing_columns:
                        alter_query = f'ALTER TABLE "{table_name}" ADD COLUMN "{column}" TEXT'
                        cursor.execute(alter_query)
                        self.table_columns[table_name].add(column)
                        logging.info(f"Added new column: {column} to table: {table_name}")
                
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e):
                        logging.error(f"Error adding column {column}: {e}")
                        raise
            
            conn.commit()
            
        except Exception as e:
            logging.error(f"Error altering table: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get schema information for a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column definitions
        """
        conn = None
        try:
            conn = self.create_connection()
            cursor = conn.cursor()
            
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            schema = []
            for col in columns:
                schema.append({
                    'cid': col[0],
                    'name': col[1],
                    'type': col[2],
                    'notnull': col[3],
                    'dflt_value': col[4],
                    'pk': col[5]
                })
            
            return schema
            
        except Exception as e:
            logging.error(f"Error getting table schema: {e}")
            raise
        finally:
            if conn:
                conn.close()



    def sanitize_table_name(self, table_name: str) -> str:
        """
        Sanitize table name by replacing invalid characters
        
        Args:
            table_name: Original table name
            
        Returns:
            Sanitized table name
        """
        # Replace periods, spaces, and hyphens with underscores
        sanitized = table_name.replace('.', '_').replace(' ', '_').replace('-', '_')
        # Remove any other invalid characters
        sanitized = ''.join(c for c in sanitized if c.isalnum() or c == '_')
        return sanitized