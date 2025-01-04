import re

# File paths provided in your format
files = {    
    "main_EA.mq5": r"C:\Users\StdUser\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Experts\ReStructured\main_EA.mq5",
    "FileLogging.mqh": r"C:\Users\StdUser\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Include\FileLogging.mqh",
    "IndicatorScoring.mqh": r"C:\Users\StdUser\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Include\IndicatorScoring.mqh",
    "TradingFunctions.mqh": r"C:\Users\StdUser\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Include\TradingFunctions.mqh"
}

# Regular expression to match function signatures in MQL5
function_pattern = re.compile(r'(\w[\w\s]+?\s+\w+\s*\([^)]*\))\s*\{')

# Function to extract and print barebone structure
def extract_barebone_structure(filename, filepath):
    try:
        # Open the file and read the content
        with open(filepath, "r") as file:
            content = file.read()
        
        # Find all function signatures
        functions = function_pattern.findall(content)
        
        # Print the structure in the desired compact format
        print(f"{filename}-------")
        for function in functions:
            print(f"{function}{{}}")
    except FileNotFoundError:
        print(f"File {filename} not found at {filepath}")
# Iterate through all files and extract structure
for filename, filepath in files.items():
    extract_barebone_structure(filename, filepath)
