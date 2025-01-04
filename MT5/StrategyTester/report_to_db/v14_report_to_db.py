import os
import re
import csv
from bs4 import BeautifulSoup

def extract_run_id_from_filename(html_file_path):
    """
    Extracts the run_id from the HTML file name.
    The expected format is <run_id>_report.html.
    
    Args:
        html_file_path (str): Path to the HTML file.

    Returns:
        str: Extracted run_id.
    """
    file_name = os.path.basename(html_file_path)
    match = re.match(r"^(.*?)_report\.html$", file_name, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Filename '{file_name}' does not match the expected format '<run_id>_report.html'")

def parse_strategy_tester_report(html_file):
    """
    Parses the Strategy Tester Report HTML file and extracts relevant data.

    Args:
        html_file (str): Path to the HTML file.

    Returns:
        dict: Extracted data categorized into 'Settings', 'Results', 'Orders', and 'Deals'.
    """
    # Read the HTML file with UTF-16 encoding
    try:
        with open(html_file, 'r', encoding='utf-16', errors='replace') as file:
            soup = BeautifulSoup(file, 'html.parser')
    except Exception as e:
        print(f"Failed to read the file with UTF-16 encoding: {e}")
        return {}

    # Find all tables in the HTML
    tables = soup.find_all('table')

    # Initialize separate lists for each section
    settings_data = []
    results_data = []
    orders_data = []
    deals_data = []

    # Define the list of fields that need to be split (Results fields)
    # Converted to lowercase for case-insensitive matching
    fields_to_split = {
        'balance drawdown maximal',
        'equity drawdown maximal',
        'balance drawdown relative',
        'equity drawdown relative',
        'z-score',
        'ahpr',
        'ghpr',
        'profit trades % of total',
        'loss trades % of total',
        'maximum consecutive wins $',
        'maximum consecutive losses $',
        'maximal consecutive profit count',
        'maximal consecutive loss count',
        'short trades won %',
        'long trades won %',
        # Add any additional Results fields here
    }

    # Convert fields_to_split to lowercase for case-insensitive matching
    fields_to_split_lower = set(label.lower() for label in fields_to_split)

    # Flag to indicate which table is being processed
    # Assuming the first table contains 'Settings' and 'Results'
    # Subsequent tables contain 'Orders' and 'Deals'
    for table_index, table in enumerate(tables, start=1):
        print(f"\n{'='*40}\nParsing Table {table_index}\n{'='*40}")
        rows = table.find_all('tr')

        # Initialize current_section based on table_index
        if table_index == 1:
            # The first table may contain both 'Settings' and 'Results'
            current_section = 'Settings/Results'
        else:
            current_section = None  # Will be set based on headers

        # Initialize flags and temporary storage for 'Inputs'
        is_collecting_inputs = False
        inputs_list = []

        for row in rows:
            # Check for section headers
            header = row.find('th')
            if header:
                header_text = header.get_text(strip=True).lower()
                if 'orders' in header_text:
                    current_section = 'Orders'
                    print("Section: Orders")
                    continue
                elif 'deals' in header_text:
                    current_section = 'Deals'
                    print("Section: Deals")
                    continue
                elif 'results' in header_text or 'strategy tester report' in header_text:
                    if table_index == 1:
                        current_section = 'Results'
                        print("Section: Results")
                    continue

            # Skip rows containing images
            if row.find('img'):
                continue

            # Find all table data cells in the row
            cells = row.find_all('td')

            # Skip rows that don't have any cells
            if not cells:
                continue

            if table_index == 1 and current_section in ['Settings/Results', 'Results']:
                if current_section == 'Settings/Results' or current_section == 'Results':
                    if is_collecting_inputs:
                        # Check if the row has a new label (i.e., the first cell ends with ':')
                        label_cell_text = cells[0].get_text(strip=True)
                        if label_cell_text.endswith(':') and label_cell_text.lower() != 'inputs:':
                            # Stop collecting 'Inputs' and save the collected data
                            if inputs_list:
                                concatenated_inputs = '; '.join(inputs_list)
                                settings_data.append(('Inputs', concatenated_inputs))
                                inputs_list = []
                            is_collecting_inputs = False

                        # If still collecting 'Inputs', extract the value from the last cell
                        if is_collecting_inputs:
                            value_cell = cells[-1]  # Assuming value is in the last cell
                            value = value_cell.get_text(strip=True)
                            if value:
                                inputs_list.append(value)
                            continue  # Move to the next row

                    # Not currently collecting 'Inputs', process normally
                    extracted_pairs = extract_key_value_pairs(cells, fields_to_split)

                    for label, value in extracted_pairs:
                        label_lower = label.lower()
                        if label_lower == 'inputs':
                            # Start collecting 'Inputs' key-value pairs
                            is_collecting_inputs = True
                            inputs_list.append(value)
                        elif label_lower in fields_to_split_lower:
                            # Assign split pairs to Results
                            split_pairs = split_specific_fields(label, value, fields_to_split_lower)
                            results_data.extend(split_pairs)
                        else:
                            # Assign to Settings
                            settings_data.append((label, value))
            elif table_index > 1 and current_section in ['Orders', 'Deals']:
                # Extract Orders and Deals data
                row_data = [cell.get_text(strip=True) for cell in cells]
                print(f"{current_section} Row: {row_data}")
                
                # Skip rows that are completely empty or contain only empty strings
                if all(cell == '' for cell in row_data):
                    continue

                # Append only if row_data has the expected number of columns
                if current_section == 'Orders':
                    # Expecting 11 columns based on headers
                    if len(row_data) >= 11:
                        orders_data.append(row_data[:11])  # Truncate to first 11 elements
                    else:
                        print(f"Skipped incomplete Orders row: {row_data}")
                elif current_section == 'Deals':
                    # Expecting 13 columns based on headers
                    if len(row_data) >= 13:
                        deals_data.append(row_data[:13])  # Truncate to first 13 elements
                    else:
                        print(f"Skipped incomplete Deals row: {row_data}")

        # After processing all rows, check if 'Inputs' were being collected
        if table_index == 1 and current_section in ['Settings/Results', 'Results'] and is_collecting_inputs and inputs_list:
            concatenated_inputs = '; '.join(inputs_list)
            settings_data.append(('Inputs', concatenated_inputs))

        print(f"Finished parsing Table {table_index}\n")

    return {
        'Settings': settings_data,
        'Results': results_data,
        'Orders': orders_data,
        'Deals': deals_data
    }

def extract_key_value_pairs(cells, fields_to_split):
    """
    Extracts all key-value pairs from a list of table cells.
    Splits pairs with labels and/or values containing parentheses into separate entries for specified fields.

    Args:
        cells (list): List of BeautifulSoup 'td' elements.
        fields_to_split (set): Set of labels that require splitting.

    Returns:
        list: List of tuples containing (label, value).
    """
    pairs = []
    i = 0
    while i < len(cells):
        cell_text = cells[i].get_text(strip=True)
        if cell_text.endswith(':'):
            label = cell_text.rstrip(':').strip()
            # Initialize value
            value = ''
            # Check if the next cell exists
            if i + 1 < len(cells):
                next_cell = cells[i + 1]
                value = next_cell.get_text(strip=True)
                # Handle cases where value might span multiple cells due to colspan
                j = i + 2
                while j < len(cells) and not cells[j].get_text(strip=True).endswith(':'):
                    value += ' ' + cells[j].get_text(strip=True)
                    j += 1
                i = j - 1  # Update i to the last processed cell
            # Now, split label and value if label is in fields_to_split
            split_pairs = split_specific_fields(label, value, fields_to_split)
            pairs.extend(split_pairs)
        i += 1
    return pairs

def split_specific_fields(label, value, fields_to_split_lower):
    """
    Splits specific labels and values containing parentheses into separate key-value pairs.

    Args:
        label (str): The original label.
        value (str): The original value.
        fields_to_split_lower (set): Set of labels that require splitting (all in lowercase).

    Returns:
        list: List of tuples with split labels and values.
    """
    label_lower = label.lower()
    if label_lower in fields_to_split_lower:
        # Regex to match values like "31 (46.97%)" or "89.47 (4)"
        value_match = re.match(r'^(.*?)\s*\((.*?)\)$', value)
        if value_match:
            main_value = value_match.group(1).strip()
            sub_value = value_match.group(2).strip()

            # Determine label suffix based on the original label
            if 'won' in label_lower:
                # Example: "Short Trades (won %)" becomes "Short Trades won" and "Short Trades won %"
                base_label = re.sub(r'\s*\(.*?\)', '', label).strip()
                main_label = f"{base_label} won"
                sub_label = f"{base_label} won %"
            elif 'profit trades' in label_lower or 'loss trades' in label_lower:
                # Example: "Profit Trades (% of total)" becomes "Profit Trades count" and "Profit Trades % of total"
                base_label = re.sub(r'\s*\(.*?\)', '', label).strip()
                main_label = f"{base_label} count"
                sub_label = f"{base_label} % of total"
            elif 'maximum consecutive wins' in label_lower or 'maximum consecutive losses' in label_lower:
                # Example: "Maximum consecutive wins ($)" becomes "Maximum consecutive wins count" and "Maximum consecutive wins value"
                base_label = re.sub(r'\s*\(.*?\)', '', label).strip()
                main_label = f"{base_label} count"
                sub_label = f"{base_label} value"
            elif 'maximal consecutive profit' in label_lower or 'maximal consecutive loss' in label_lower:
                # Example: "Maximal consecutive profit (count)" becomes "Maximal consecutive profit value" and "Maximal consecutive profit count"
                base_label = re.sub(r'\s*\(.*?\)', '', label).strip()
                main_label = f"{base_label} value"
                sub_label = f"{base_label} count"
            elif 'drawdown' in label_lower or 'z-score' in label_lower or 'ahpr' in label_lower or 'ghpr' in label_lower:
                # Example: "Balance Drawdown Maximal" becomes "Balance Drawdown Maximal value" and "Balance Drawdown Maximal percentage"
                if 'relative' in label_lower:
                    main_label = f"{label} percentage"
                    sub_label = f"{label} amount"
                else:
                    main_label = f"{label} value"
                    sub_label = f"{label} percentage"
            else:
                # Default splitting
                main_label = f"{label} value"
                sub_label = f"{label} percentage"

            # Handle special cases based on label structure
            if 'relative' in label_lower:
                # For 'Balance Drawdown Relative' etc.
                main_label = f"{label} percentage"
                sub_label = f"{label} amount"

            # Return the split pairs
            return [
                (main_label, main_value),
                (sub_label, sub_value)
            ]
        else:
            # If value doesn't have '(', return as is
            return [(label, value)]
    else:
        # For labels not in fields_to_split, return as is
        return [(label, value)]

def save_to_csv(data, filename):
    """
    Saves the provided data to a CSV file with additional debugging.

    Args:
        data (list of tuples or lists): Data to be saved.
        filename (str): Name of the CSV file.
    """
    if not data:
        print(f"No data to write to '{filename}'. Skipping file creation.")
        return

    try:
        with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            rows_written = 0

            if filename.startswith(('settings', 'results')):
                writer.writerow(['Label', 'Value'])
                for row in data:
                    if isinstance(row, (list, tuple)) and len(row) == 2:
                        writer.writerow(row)
                        rows_written += 1
                    else:
                        print(f"Skipping invalid row in {filename}: {row}")
            elif filename.startswith(('orders', 'deals')):
                if data:
                    writer.writerow(data[0])
                    rows_written += 1
                    for row in data[1:]:
                        if len(row) == len(data[0]):
                            writer.writerow(row)
                            rows_written += 1
                        else:
                            print(f"Skipping invalid row in {filename}: {row}")

        print(f"Data successfully saved to '{filename}'. Rows written: {rows_written}")
        
        # Verify file content
        with open(filename, 'r', encoding='utf-8-sig') as csvfile:
            content = csvfile.read()
            print(f"File content (first 200 characters): {content[:200]}")
            
    except Exception as e:
        print(f"Failed to save data to '{filename}': {e}")
        raise



def save_settings_and_results_to_csv(data, run_id):
    """
    Saves the Settings and Results data into separate CSV files with the run_id as prefix.

    Args:
        data (dict): Dictionary containing 'Settings' and 'Results'.
        run_id (str): The run ID to prefix the CSV filenames.
    """
    try:
        if data.get('Settings'):
            print(f"Saving {len(data['Settings'])} Settings rows")
            save_to_csv(data['Settings'], f'{run_id}_settings_data.csv')
        else:
            print("No Settings data to save.")

        if data.get('Results'):
            print(f"Saving {len(data['Results'])} Results rows")
            save_to_csv(data['Results'], f'{run_id}_results_data.csv')
        else:
            print("No Results data to save.")
    except Exception as e:
        print(f"Error saving Settings and Results: {e}")
        raise

def save_orders_and_deals_to_csv(data, run_id):
    """
    Saves the Orders and Deals data into separate CSV files with the run_id as prefix.

    Args:
        data (dict): Dictionary containing 'Orders' and 'Deals'.
        run_id (str): The run ID to prefix the CSV filenames.
    """
    try:
        if data.get('Orders'):
            print(f"Saving {len(data['Orders'])} Orders rows")
            save_to_csv(data['Orders'], f'{run_id}_orders_data.csv')
        else:
            print("No Orders data to save.")

        if data.get('Deals'):
            print(f"Saving {len(data['Deals'])} Deals rows")
            save_to_csv(data['Deals'], f'{run_id}_deals_data.csv')
        else:
            print("No Deals data to save.")
    except Exception as e:
        print(f"Error saving Orders and Deals: {e}")
        raise


if __name__ == "__main__":
    html_file_path = r"C:\Users\StdUser\Desktop\MyProjects\Backtesting\logs\AMD_10021629_report.html"

    if not os.path.exists(html_file_path):
        print(f"HTML file '{html_file_path}' does not exist. Please check the path.")
    else:
        try:
            run_id = extract_run_id_from_filename(html_file_path)
            print(f"Extracted run_id: {run_id}")

            data = parse_strategy_tester_report(html_file_path)

            print("\nExtracted Data Structure:")
            for key, value in data.items():
                print(f"{key}: {type(value)} with {len(value)} items")
                if value:
                    print(f"Sample of first item: {value[0]}")

            try:
                save_settings_and_results_to_csv(data, run_id)
            except Exception as e:
                print(f"Error in saving Settings and Results: {e}")

            try:
                save_orders_and_deals_to_csv(data, run_id)
            except Exception as e:
                print(f"Error in saving Orders and Deals: {e}")

            print("\nChecking saved CSV files:")
            for file_type in ['settings', 'results', 'orders', 'deals']:
                filename = f"{run_id}_{file_type}_data.csv"
                if os.path.exists(filename):
                    file_size = os.path.getsize(filename)
                    print(f"{filename}: {file_size} bytes")
                    if file_size == 0:
                        print(f"Warning: {filename} is empty!")
                    else:
                        with open(filename, 'r', encoding='utf-8-sig') as f:
                            print(f"First line of {filename}: {f.readline().strip()}")
                else:
                    print(f"{filename} not found!")

        except Exception as e:
            print(f"An error occurred during processing: {e}")
            raise