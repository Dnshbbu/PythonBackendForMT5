import sqlite3
import pandas as pd

def check_predictions():
    # Connect to the database
    db_path = "logs/trading_data.db"
    conn = sqlite3.connect(db_path)
    
    try:
        # Query the most recent predictions
        query = """
        SELECT datetime, actual_price, predicted_price, error, model_name, source_table
        FROM historical_predictions
        ORDER BY datetime DESC
        LIMIT 5
        """
        
        # Read into a pandas DataFrame
        df = pd.read_sql_query(query, conn)
        
        # Display the results
        print("\nMost recent predictions:")
        print(df)
        
        # Get total count
        count_query = "SELECT COUNT(*) FROM historical_predictions"
        count = pd.read_sql_query(count_query, conn).iloc[0, 0]
        print(f"\nTotal number of predictions stored: {count}")
        
    finally:
        conn.close()

if __name__ == "__main__":
    check_predictions() 