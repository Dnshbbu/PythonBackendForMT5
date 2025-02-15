import sqlite3
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the current directory and construct database path
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, 'logs', 'trading_data.db')

logging.info(f"Using database at: {db_path}")
logging.info(f"Database exists: {os.path.exists(db_path)}")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    if not tables:
        logging.info("No tables found in the database")
    else:
        logging.info("\nTables in database:")
        for table in tables:
            table_name = table[0]
            logging.info(f"- {table_name}")
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            logging.info(f"\nSchema for {table_name}:")
            for col in columns:
                logging.info(f"  - {col[1]} ({col[2]})")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            logging.info(f"Row count: {count}")
            
            # Show a sample row if table has data and is related to models
            if count > 0 and ('model' in table_name.lower() or 'repository' in table_name.lower()):
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
                sample = cursor.fetchone()
                logging.info("Sample row:")
                for col, val in zip([c[1] for c in columns], sample):
                    logging.info(f"  {col}: {val}")
    
except sqlite3.Error as e:
    logging.error(f"Database error: {e}")
except Exception as e:
    logging.error(f"Error: {str(e)}")
finally:
    if 'conn' in locals():
        conn.close() 