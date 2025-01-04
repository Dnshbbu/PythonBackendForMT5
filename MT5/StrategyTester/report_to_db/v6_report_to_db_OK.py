from bs4 import BeautifulSoup
import csv
import os

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

    # Flag to indicate which table is being processed
    # Assuming the first table contains 'Settings' and 'Results'
    # Subsequent tables contain 'Orders' and 'Deals'
    for table_index, table in enumerate(tables, start=1):
        print(f"\n{'='*40}\nParsing Table {table_index}\n{'='*40}")
        rows = table.find_all('tr')

        # Initialize current_section based on table_index
        if table_index == 1:
            current_section = 'Settings'
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

            if table_index == 1 and current_section in ['Settings', 'Results']:
                if current_section == 'Settings':
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
                    extracted_pairs = extract_key_value_pairs(cells)

                    for label, value in extracted_pairs:
                        if label == 'Inputs':
                            # Start collecting 'Inputs' key-value pairs
                            is_collecting_inputs = True
                            inputs_list.append(value)
                        else:
                            settings_data.append((label, value))
                elif current_section == 'Results':
                    # Extract key-value pairs for Results
                    extracted_pairs = extract_key_value_pairs(cells)
                    results_data.extend(extracted_pairs)

            elif table_index > 1 and current_section in ['Orders', 'Deals']:
                # Extract Orders and Deals data
                row_data = [cell.get_text(strip=True) for cell in cells]
                print(f"{current_section} Row: {row_data}")
                if current_section == 'Orders':
                    orders_data.append(row_data)
                elif current_section == 'Deals':
                    deals_data.append(row_data)

        # After processing all rows, check if 'Inputs' were being collected
        if table_index == 1 and current_section == 'Settings' and is_collecting_inputs and inputs_list:
            concatenated_inputs = '; '.join(inputs_list)
            settings_data.append(('Inputs', concatenated_inputs))

        print(f"Finished parsing Table {table_index}\n")

    return {
        'Settings': settings_data,
        'Results': results_data,
        'Orders': orders_data,
        'Deals': deals_data
    }

def extract_key_value_pairs(cells):
    """
    Extracts all key-value pairs from a list of table cells.

    Args:
        cells (list): List of BeautifulSoup 'td' elements.

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
            pairs.append((label, value))
        i += 1
    return pairs

def save_to_csv(data, filename):
    """
    Saves the provided data to a CSV file.

    Args:
        data (list of tuples or lists): Data to be saved.
        filename (str): Name of the CSV file.
    """
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if filename.startswith('settings') or filename.startswith('results'):
                writer.writerow(['Label', 'Value'])
                writer.writerows(data)
            elif filename.startswith('orders'):
                # Define headers based on the Orders table structure
                headers = ['Open Time', 'Order', 'Symbol', 'Type', 'Volume', 'Price', 'S/L', 'T/P', 'Time', 'State', 'Comment']
                writer.writerow(headers)
                writer.writerows(data)
            elif filename.startswith('deals'):
                # Define headers based on the Deals table structure
                headers = ['Time', 'Deal', 'Symbol', 'Type', 'Direction', 'Volume', 'Price', 'Order', 'Commission', 'Swap', 'Profit', 'Balance', 'Comment']
                writer.writerow(headers)
                writer.writerows(data)
        print(f"Data successfully saved to '{filename}'.")
    except Exception as e:
        print(f"Failed to save data to '{filename}': {e}")

def save_settings_and_results_to_csv(data):
    """
    Saves the Settings and Results data into separate CSV files.

    Args:
        data (dict): Dictionary containing 'Settings' and 'Results'.
    """
    # Save Settings
    if data.get('Settings'):
        save_to_csv(data['Settings'], 'settings_data.csv')

    # Save Results
    if data.get('Results'):
        save_to_csv(data['Results'], 'results_data.csv')

def save_orders_and_deals_to_csv(data):
    """
    Saves the Orders and Deals data into separate CSV files.

    Args:
        data (dict): Dictionary containing 'Orders' and 'Deals'.
    """
    # Save Orders
    if data.get('Orders'):
        save_to_csv(data['Orders'], 'orders_data.csv')

    # Save Deals
    if data.get('Deals'):
        save_to_csv(data['Deals'], 'deals_data.csv')

if __name__ == "__main__":
    # Replace 'report.html' with the path to your HTML file
    html_file_path = r"C:\Users\StdUser\Desktop\MyProjects\Backtesting\logs\AMD_10021629_report.html"  # Replace with your file path

    # Check if the HTML file exists
    if not os.path.exists(html_file_path):
        print(f"HTML file '{html_file_path}' does not exist. Please check the path.")
    else:
        # Parse the HTML report
        data = parse_strategy_tester_report(html_file_path)

        # Save Settings and Results to CSV
        save_settings_and_results_to_csv(data)

        # Save Orders and Deals to CSV
        save_orders_and_deals_to_csv(data)

        # Optionally, print all extracted data for verification
        print("\nExtracted Data:")

        # Print Settings
        print("\n--- Settings ---")
        for label, value in data.get('Settings', []):
            print(f"{label}: {value}")

        # Print Results
        print("\n--- Results ---")
        for label, value in data.get('Results', []):
            print(f"{label}: {value}")

        # Print Orders
        print("\n--- Orders ---")
        for order in data.get('Orders', []):
            print(order)

        # Print Deals
        print("\n--- Deals ---")
        for deal in data.get('Deals', []):
            print(deal)
