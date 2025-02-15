import sqlite3
import argparse
from typing import List, Dict, Optional

def get_table_schema(db_path: str, table_name: str) -> List[Dict]:
    """
    Get schema information for a specific table in SQLite database.
    
    Args:
        db_path: Path to the SQLite database
        table_name: Name of the table to get schema for
        
    Returns:
        List of dictionaries containing column information:
        [
            {
                'column_name': str,
                'data_type': str,
                'is_nullable': bool,
                'default_value': Optional[str],
                'is_primary_key': bool
            },
            ...
        ]
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute(f"""
            SELECT name 
            FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (table_name,))
        
        if not cursor.fetchone():
            raise ValueError(f"Table '{table_name}' does not exist in the database")
        
        # Get table info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        schema = []
        for col in columns:
            cid, name, type_, not_null, default_value, pk = col
            schema.append({
                'column_name': name,
                'data_type': type_,
                'is_nullable': not not_null,
                'default_value': default_value,
                'is_primary_key': bool(pk)
            })
        
        return schema
        
    except Exception as e:
        raise Exception(f"Error getting schema: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

def print_schema(schema: List[Dict]):
    """Print schema in a readable format"""
    print("\nTable Schema:")
    print("-" * 80)
    print(f"{'Column Name':<20} {'Data Type':<15} {'Nullable':<10} {'Default':<15} {'Primary Key':<10}")
    print("-" * 80)
    
    for col in schema:
        print(
            f"{col['column_name']:<20} "
            f"{col['data_type']:<15} "
            f"{'Yes' if col['is_nullable'] else 'No':<10} "
            f"{str(col['default_value']):<15} "
            f"{'Yes' if col['is_primary_key'] else 'No':<10}"
        )
    print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description='Get schema for a SQLite database table')
    parser.add_argument('db_path', help='Path to the SQLite database')
    parser.add_argument('table_name', help='Name of the table to get schema for')
    
    args = parser.parse_args()
    
    try:
        schema = get_table_schema(args.db_path, args.table_name)
        print_schema(schema)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 