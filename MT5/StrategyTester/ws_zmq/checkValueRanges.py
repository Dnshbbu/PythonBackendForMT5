# import pandas as pd

# def check_value_range(value):
#     try:
#         # Convert to float and check range
#         num_value = float(value)
#         return -100 <= num_value <= 100
#     except (ValueError, TypeError):
#         # Return True for non-numeric values to skip them
#         return True

# def analyze_columns(df, columns_to_check):
#     issues = []
    
#     for column in columns_to_check:
#         for idx, cell in enumerate(df[column]):
#             # Skip if cell is empty or NaN
#             if pd.isna(cell) or cell == '':
#                 continue
                
#             # Split the cell into name-value pairs
#             pairs = cell.split('|')
            
#             for pair in pairs:
#                 if '=' not in pair:
#                     continue
                    
#                 name, value = pair.split('=')
                
#                 # Check if value is outside range
#                 if not check_value_range(value):
#                     issues.append({
#                         'row': idx,
#                         'column': column,
#                         'name': name,
#                         'value': value
#                     })
    
#     return issues

# # Usage
# def main():
#     # Read the CSV file
#     df = pd.read_csv(r"C:\Users\StdUser\Desktop\MyProjects\Backtesting\logs\SYM_10027109_transactions.csv")
    
#     # Columns to check
#     columns_to_check = ['factors', 'score', 'efactors', 'exitScore']
    
#     # Analyze the columns
#     issues = analyze_columns(df, columns_to_check)
    
#     # Print the issues
#     if issues:
#         print("Found values outside the range -100 to +100:")
#         for issue in issues:
#             print(f"Row {issue['row']}, Column: {issue['column']}, "
#                   f"Name: {issue['name']}, Value: {issue['value']}")
#     else:
#         print("No values found outside the range -100 to +100")

# if __name__ == "__main__":
#     main()


import pandas as pd
import os

def check_value_range(value):
    try:
        # Convert to float and check range
        num_value = float(value)
        return -100 <= num_value <= 100
    except (ValueError, TypeError):
        # Return True for non-numeric values to skip them
        return True

def analyze_columns(df, columns_to_check):
    issues = []
    
    for column in columns_to_check:
        for idx, cell in df[column].items():  # Using items() instead of enumerate
            # Skip if cell is empty or NaN
            if pd.isna(cell) or cell == '':
                continue
                
            # Split the cell into name-value pairs
            pairs = cell.split('|')
            
            for pair in pairs:
                if '=' not in pair:
                    continue
                    
                name, value = pair.split('=')
                
                # Check if value is outside range
                if not check_value_range(value):
                    issues.append({
                        'row': idx + 2,  # Adding 2: 1 for header and 1 because Excel starts at 1
                        'column': column,
                        'name': name,
                        'value': value
                    })
    
    return issues

# Usage
def main():
    filepath = r"C:\Users\StdUser\Desktop\MyProjects\Backtesting\logs\SYM_10021860_transactions.csv"
    filename = os.path.basename(filepath)  # Gets just the file name without the path
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Columns to check
    columns_to_check = ['factors', 'score', 'efactors', 'exitScore']
    
    # Analyze the columns
    issues = analyze_columns(df, columns_to_check)
    
    # Print the issues
    if issues:
        print(f"Analyzing file: {filename}")
        print("Found values outside the range -100 to +100:")
        for issue in issues:
            print(f"Row {issue['row']}, Column: {issue['column']}, "
                  f"Name: {issue['name']}, Value: {issue['value']}")
    else:
        print("No values found outside the range -100 to +100")

if __name__ == "__main__":
    main()