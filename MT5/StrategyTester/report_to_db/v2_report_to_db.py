import re
import pandas as pd
from bs4 import BeautifulSoup

def parse_strategy_tester_report(html_file_path):
    # Read the HTML report file and parse it using BeautifulSoup
    with open(html_file_path, 'r', encoding='utf-16') as file:
        report_text = file.read()

    # Parse the report to extract settings and results
    data = parse_report(report_text)

    return data

def parse_report(report_text):
    # Dictionary to hold extracted data
    data = {
        'Settings': {},
        'Results': {}
    }

    # Split the report into sections
    sections = report_text.split('\n\n')

    # Define regex patterns for Inputs and Results sections
    input_pattern = r'^(Expert|Symbol|Period|Inputs):\s*(.*)$'  # Matches "Name: Value" format
    result_pattern = r'^(.*?):\s*(.*)$'  # Matches "Name: Value" format in Results

    # Extract Inputs from Settings
    settings_section = sections[0] if len(sections) > 0 else ""
    lines = settings_section.split('\n')
    
    # To capture multiple inputs correctly
    input_started = False
    for line in lines:
        match = re.match(input_pattern, line)
        if match:
            name, value = match.groups()
            if name == 'Inputs':
                input_started = True  # Start capturing inputs
                data['Settings']['Inputs'] = [value]  # Initialize list with the first input
            else:
                # Store other settings
                data['Settings'][name] = value
        elif input_started and line.strip():  # Continue capturing inputs if we are in the Inputs section
            data['Settings']['Inputs'].append(line.strip())

    # Extract Results
    results_section = sections[1] if len(sections) > 1 else ""
    for line in results_section.split('\n'):
        match = re.match(result_pattern, line)
        if match:
            name, value = match.groups()
            data['Results'][name] = value

    return data

# Example usage
html_file_path = r"C:/Users/StdUser/Desktop/MyProjects/Backtesting/logs/AMD_10021629_report.html"
parsed_data = parse_strategy_tester_report(html_file_path)

# Convert parsed data to DataFrame
settings_df = pd.DataFrame.from_dict(parsed_data['Settings'], orient='index', columns=['Value'])
results_df = pd.DataFrame.from_dict(parsed_data['Results'], orient='index', columns=['Value'])

# Display the DataFrames
print("Settings:")
print(settings_df)
print("\nResults:")
print(results_df)