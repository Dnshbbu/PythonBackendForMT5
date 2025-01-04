# database.py

import psycopg2
from psycopg2 import sql
import datetime
import uuid
import os
from config import DATABASE_CONFIG
import logging

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def connect_to_db():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=DATABASE_CONFIG['host'],
            dbname=DATABASE_CONFIG['dbname'],
            user=DATABASE_CONFIG['user'],
            password=DATABASE_CONFIG['password'],
            port=DATABASE_CONFIG['port']
        )
        logging.info("Database connection established.")
        return conn
    except psycopg2.Error as e:
        logging.error(f"Database connection failed: {e}")
        raise

def create_performance_table():
    """Creates the 'runs_performance' table if it doesn't exist."""
    conn = connect_to_db()
    cursor = conn.cursor()

    create_table_query = """
    CREATE TABLE IF NOT EXISTS runs_performance (
        Datetime TIMESTAMP,
        Run_id UUID PRIMARY KEY,
        Final_strategy_id TEXT,
        File_executed TEXT,
        Strategy TEXT,
        Run_number INTEGER,
        Run_Parameters TEXT,
        All_parameters TEXT,
        Confidence_level REAL,
        Stock TEXT,
        Start_date DATE,
        End_date DATE,
        Start_price REAL,
        End_price REAL,
        Initial_balance REAL,
        Final_balance REAL,
        Returns REAL,
        Profit_Loss REAL,
        Success_rate REAL,
        Trades_executed INTEGER,
        Successful INTEGER,
        Failed INTEGER,
        Success_reason TEXT,
        Failure_reason TEXT,
        Optimized BOOLEAN,
        Influencers TEXT,
        Catalyst TEXT,
        Chart_html TEXT,
        Comments TEXT
    );
    """

    try:
        cursor.execute(create_table_query)
        conn.commit()
        logging.info("Table 'runs_performance' ensured in database.")
    except psycopg2.Error as e:
        logging.error(f"Failed to create table: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

def save_run_results_to_db(run_data, stock, start_date, end_date, chart_html, strategy_class_name):
    """Inserts backtest results into the 'runs_performance' table."""
    conn = connect_to_db()
    cursor = conn.cursor()

    insert_query = """
    INSERT INTO runs_performance (
        Datetime, Run_id, Final_strategy_id, File_executed, Strategy, Run_number, Run_Parameters, 
        All_parameters, Confidence_level, Stock, Start_date, End_date, Start_price, 
        End_price, Initial_balance, Final_balance, Returns, Profit_Loss, 
        Success_rate, Trades_executed, Successful, Failed, Success_reason, 
        Failure_reason, Optimized, Influencers, Catalyst, Chart_html, Comments
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s)
    """

    current_file_name = os.path.basename(__file__)
    run_id = str(uuid.uuid4())

    successful = 1 if run_data['profit_loss'] > 0 else 0
    failed = 1 if run_data['profit_loss'] <= 0 else 0
    success_reason = "Profit" if successful else "N/A"
    failure_reason = "Loss" if failed else "N/A"

    data_to_insert = (
        datetime.datetime.now(),
        run_id,
        'N/A',  # Example strategy ID
        current_file_name,
        strategy_class_name,
        1,        # Run_number
        'N/A',    # Run_Parameters
        'N/A',    # All_parameters
        0.95,     # Confidence_level
        stock,
        start_date,
        end_date,
        run_data['initial_balance'],
        run_data['final_balance'],
        run_data['initial_balance'],
        run_data['final_balance'],
        run_data['returns'],
        run_data['profit_loss'],
        1.0,      # Success_rate
        1,        # Trades_executed
        successful,
        failed,
        success_reason,
        failure_reason,
        False,    # Optimized
        'N/A',    # Influencers
        'N/A',    # Catalyst
        chart_html,
        'N/A'     # Comments
    )

    try:
        cursor.execute(insert_query, data_to_insert)
        conn.commit()
        logging.info("Backtest results saved to database.")
    except psycopg2.Error as e:
        logging.error(f"Failed to insert run results: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()
