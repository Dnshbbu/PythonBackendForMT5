import sqlite3
import json

def check_model_info(model_name):
    try:
        conn = sqlite3.connect('logs/trading_data.db')
        cursor = conn.cursor()
        
        # Get column names
        cursor.execute('PRAGMA table_info(model_repository)')
        columns = [col[1] for col in cursor.fetchall()]
        
        # Get model info
        cursor.execute('SELECT * FROM model_repository WHERE model_name = ?', (model_name,))
        row = cursor.fetchone()
        
        if row:
            model_info = dict(zip(columns, row))
            
            # Parse JSON fields
            json_fields = ['feature_importance', 'model_params', 'metrics', 'additional_metadata']
            for field in json_fields:
                if model_info[field]:
                    model_info[field] = json.loads(model_info[field])
            
            print("\nModel Repository Information:")
            print("-" * 50)
            for key, value in model_info.items():
                if isinstance(value, (dict, list)):
                    print(f"\n{key}:")
                    print(json.dumps(value, indent=2))
                else:
                    print(f"{key}: {value}")
        else:
            print(f"No information found for model: {model_name}")
            
        conn.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    check_model_info('ts_sarima_test') 