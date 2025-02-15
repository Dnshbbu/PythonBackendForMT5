import sqlite3
import os

def clean_null_timestamps():
    """Delete rows with NULL timestamps from historical predictions tables"""
    db_path = os.path.join('logs', 'trading_data.db')
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # First, get the run_ids with NULL run_timestamp
        cursor.execute('''
            SELECT DISTINCT run_id 
            FROM historical_prediction_metrics 
            WHERE run_timestamp IS NULL
        ''')
        null_run_ids = [row[0] for row in cursor.fetchall()]
        
        if null_run_ids:
            # Delete from historical_predictions for these run_ids
            placeholders = ','.join('?' * len(null_run_ids))
            cursor.execute(f'''
                DELETE FROM historical_predictions 
                WHERE run_id IN ({placeholders})
            ''', null_run_ids)
            predictions_deleted = cursor.rowcount
            
            # Delete from historical_prediction_metrics for these run_ids
            cursor.execute(f'''
                DELETE FROM historical_prediction_metrics 
                WHERE run_id IN ({placeholders})
            ''', null_run_ids)
            metrics_deleted = cursor.rowcount
            
            print(f'Found {len(null_run_ids)} runs with NULL timestamps')
            print(f'Run IDs with NULL timestamps: {null_run_ids}')
            print(f'Deleted {predictions_deleted} rows from historical_predictions')
            print(f'Deleted {metrics_deleted} rows from historical_prediction_metrics')
            
            # Commit changes
            conn.commit()
            print('Changes committed successfully')
        else:
            print('No runs found with NULL timestamps')
        
    except sqlite3.Error as e:
        print(f'Error: {str(e)}')
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    clean_null_timestamps() 