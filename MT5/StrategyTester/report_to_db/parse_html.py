
# Python code
import os
import re
import csv
import uuid
import psycopg2
from psycopg2 import sql
from bs4 import BeautifulSoup
import datetime


def parse_strategy_tester_report(html_file):
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
    fields_to_split = {
        'Balance Drawdown Maximal',
        'Equity Drawdown Maximal',
        'Balance Drawdown Relative',
        'Equity Drawdown Relative',
        'Z-Score',
        'AHPR',
        'GHPR',
        'Profit Trades (% of total)',
        'Loss Trades (% of total)',
        'Maximum consecutive wins ($)',
        'Maximum consecutive losses ($)',
        'Maximal consecutive profit (count)',
        'Maximal consecutive loss (count)',
        'Short Trades (won %)',
        'Long Trades (won %)',
        # Add any additional Results fields here
    }

    # Flag to indicate which table is being processed
    # Assuming the first table contains 'Settings' and 'Results'
    # Subsequent tables contain 'Orders' and 'Deals'
    for table_index, table in enumerate(tables, start=1):
        # print(f"\n{'='*40}\nParsing Table {table_index}\n{'='*40}")
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
                    # print("Section: Orders")
                    continue
                elif 'deals' in header_text:
                    current_section = 'Deals'
                    # print("Section: Deals")
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

            # **NEW: Skip header rows in Orders and Deals**
            if current_section in ['Orders', 'Deals']:
                first_cell_text = cells[0].get_text(strip=True).lower()
                # Define possible header identifiers
                header_identifiers = {
                    'Orders': ['open time', 'order', 'symbol', 'type', 'volume', 'price', 's / l', 't / p', 'time', 'state', 'comment'],
                    'Deals': ['time', 'deal', 'symbol', 'type', 'direction', 'volume', 'price', 'order', 'commission', 'swap', 'profit', 'balance', 'comment']
                }
                expected_headers = header_identifiers.get(current_section, [])
                if first_cell_text in [h.lower() for h in expected_headers]:
                    # print(f"Skipping header row in {current_section} section.")
                    continue  # Skip header row

            if table_index == 1 and current_section in ['Settings/Results', 'Results']:
                if current_section == 'Settings/Results':
                    # Handle 'Settings/Results' combined section
                    if is_collecting_inputs:
                        # Check if the row has a new label (i.e., the first cell ends with ':')
                        label_cell_text = cells[0].get_text(strip=True)
                        if label_cell_text.endswith(':') and label_cell_text != 'Inputs:':
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
                        if label == 'Inputs':
                            # Start collecting 'Inputs' key-value pairs
                            is_collecting_inputs = True
                            inputs_list.append(value)
                        elif label in fields_to_split:
                            # Assign split pairs to Results
                            split_pairs = split_specific_fields(label, value, fields_to_split)
                            results_data.extend(split_pairs)
                        else:
                            # Assign to Settings
                            settings_data.append((label, value))
                elif current_section == 'Results':
                    # Extract key-value pairs for Results
                    extracted_pairs = extract_key_value_pairs(cells, fields_to_split)
                    results_data.extend(extracted_pairs)

            elif table_index > 1 and current_section in ['Orders', 'Deals']:
                # Extract Orders and Deals data
                row_data = [cell.get_text(strip=True) for cell in cells]
                # print(f"{current_section} Row: {row_data}")
                
                # Skip rows that are completely empty or contain only empty strings
                if all(cell == '' for cell in row_data):
                    continue

                # Append only if row_data has the expected number of columns
                if current_section == 'Orders':
                    # Expecting 11 columns based on headers
                    if len(row_data) >= 11:
                        orders_data.append(row_data[:11])  # Truncate to first 11 elements
                elif current_section == 'Deals':
                    # Expecting 13 columns based on headers
                    if len(row_data) >= 13:
                        deals_data.append(row_data[:13])  # Truncate to first 13 elements

        # After processing all rows, check if 'Inputs' were being collected
        if table_index == 1 and current_section in ['Settings/Results', 'Results'] and is_collecting_inputs and inputs_list:
            concatenated_inputs = '; '.join(inputs_list)
            settings_data.append(('Inputs', concatenated_inputs))

    # Convert percentage values to decimals in results_data
    # Convert percentage values to decimals in results_data
    for index, (label, value) in enumerate(settings_data):
        if 'percentage' in label.lower() or '%' in label:
            value = convert_percentage_to_decimal(value)
            settings_data[index] = (label, value)  # Update the value in the list

    # print(f"Finished parsing Table {table_index}\n")
    # print("================================")
    # print(settings_data)
    # print("================================")

    return {
        'Settings': settings_data,
        'Results': results_data,
        'Orders': orders_data,
        'Deals': deals_data
    }


 # Helper function to convert percentage strings to decimal

def convert_percentage_to_decimal(value):
    if isinstance(value, str) and value.endswith('%'):
        try:
            return float(value[:-1]) / 100
        except ValueError:
            return value  # Return original value if conversion fails
    return value

def extract_key_value_pairs(cells, fields_to_split):
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

def split_specific_fields(label, value, fields_to_split):
    if label in fields_to_split:
        # Regex to match values like "31 (46.97%)" or "89.47 (4)"
        value_match = re.match(r'^(.*?)\s*\((.*?)\)$', value)
        if value_match:
            main_value = value_match.group(1).strip()
            sub_value = value_match.group(2).strip()

            # Determine label suffix based on the original label
            if 'won' in label.lower():
                # Example: "Short Trades (won %)" becomes "Short Trades won" and "Short Trades won %"
                main_label = re.sub(r'\s*\(.*?\)', '', label).strip() + ' won'
                sub_label = re.sub(r'\s*\(.*?\)', '', label).strip() + ' won %'
            elif 'Profit Trades' in label or 'Loss Trades' in label:
                # Example: "Profit Trades (% of total)" becomes "Profit Trades count" and "Profit Trades % of total"
                main_label = re.sub(r'\s*\(.*?\)', '', label).strip() + ' count'
                sub_label = re.sub(r'\s*\(.*?\)', '', label).strip() + ' % of total'
            elif 'Maximum consecutive wins' in label or 'Maximum consecutive losses' in label:
                # Example: "Maximum consecutive wins ($)" becomes "Maximum consecutive wins count" and "Maximum consecutive wins value"
                main_label = re.sub(r'\s*\(.*?\)', '', label).strip() + ' count'
                sub_label = re.sub(r'\s*\(.*?\)', '', label).strip() + ' value'
            elif 'Maximal consecutive profit' in label or 'Maximal consecutive loss' in label:
                # Example: "Maximal consecutive profit (count)" becomes "Maximal consecutive profit value" and "Maximal consecutive profit count"
                main_label = re.sub(r'\s*\(.*?\)', '', label).strip() + ' value'
                sub_label = re.sub(r'\s*\(.*?\)', '', label).strip() + ' count'
            elif 'drawdown' in label.lower() or 'z-score' in label.lower() or 'ahpr' in label.lower() or 'ghpr' in label.lower():
                # Example: "Balance Drawdown Maximal" becomes "Balance Drawdown Maximal value" and "Balance Drawdown Maximal percentage"
                if 'Relative' in label:
                    main_label = label + ' percentage'
                    sub_label = label + ' amount'
                else:
                    main_label = label + ' value'
                    sub_label = label + ' percentage'
            else:
                # Default splitting
                main_label = label + ' value'
                sub_label = label + ' percentage'

            # Handle special cases based on label structure
            if 'relative' in label.lower():
                # For 'Balance Drawdown Relative' etc.
                main_label = label + ' percentage'
                sub_label = label + ' amount'

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

def sanitize_column_name(label):
    """
    Sanitizes the column name by converting to lowercase, replacing spaces and slashes with underscores,
    and removing any non-alphanumeric characters except underscores.
    
    :param label: The original label string.
    :return: A sanitized column name string.
    """
    column = label.lower()
    column = re.sub(r'[ /]+', '_', column)
    column = re.sub(r'[^a-z0-9_]', '', column)
    return column
