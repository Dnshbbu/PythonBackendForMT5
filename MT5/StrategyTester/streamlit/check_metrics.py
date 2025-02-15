import sqlite3
import pandas as pd

def check_metrics():
    # Connect to the database
    db_path = "logs/trading_data.db"
    conn = sqlite3.connect(db_path)
    
    try:
        # Query the most recent metrics
        query = """
        SELECT timestamp, model_name, source_table,
               total_predictions, mean_absolute_error, root_mean_squared_error,
               mean_absolute_percentage_error, r_squared, direction_accuracy
        FROM historical_prediction_metrics
        ORDER BY timestamp DESC
        LIMIT 5
        """
        
        # Read into a pandas DataFrame
        df = pd.read_sql_query(query, conn)
        
        # Display the results
        print("\nMost recent prediction metrics:")
        print(df)
        
        # Get total count
        count_query = "SELECT COUNT(*) FROM historical_prediction_metrics"
        count = pd.read_sql_query(count_query, conn).iloc[0, 0]
        print(f"\nTotal number of metric records stored: {count}")
        
    finally:
        conn.close()

if __name__ == "__main__":
    check_metrics() 