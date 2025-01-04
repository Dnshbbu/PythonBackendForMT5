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
    try:
        with open(html_file, 'r', encoding='utf-16', errors='replace') as file:
            soup = BeautifulSoup(file, 'html.parser')
    except Exception as e:
        print(f"Failed to read the file with UTF-16 encoding: {e}")
        return {}

    tables = soup.find_all('table')

    settings_data = []
    results_data = []
    orders_data = []
    deals_data = []

    for table_index, table in enumerate(tables, start=1):
        print(f"\n{'='*40}\nParsing Table {table_index}\n{'='*40}")
        rows = table.find_all('tr')

        current_section = 'Settings' if table_index == 1 else None

        for row in rows:
            header = row.find('th')
            if header:
                header_text = header.get_text(strip=True).lower()
                if 'orders' in header_text:
                    current_section = 'Orders'
                elif 'deals' in header_text:
                    current_section = 'Deals'
                elif 'results' in header_text or 'strategy tester report' in header_text:
                    if table_index == 1:
                        current_section = 'Results'
                continue

            if row.find('img'):
                continue

            cells = row.find_all('td')

            if not cells:
                continue

            if table_index == 1 and current_section in ['Settings', 'Results']:
                extracted_pairs = extract_key_value_pairs(cells, current_section)
                if current_section == 'Settings':
                    settings_data.extend(extracted_pairs)
                elif current_section == 'Results':
                    results_data.extend(extracted_pairs)
            elif table_index > 1 and current_section in ['Orders', 'Deals']:
                row_data = [cell.get_text(strip=True) for cell in cells]
                if current_section == 'Orders':
                    orders_data.append(row_data)
                elif current_section == 'Deals':
                    deals_data.append(row_data)

    return {
        'Settings': settings_data,
        'Results': results_data,
        'Orders': orders_data,
        'Deals': deals_data
    }

def extract_key_value_pairs(cells, section):
    """
    Extracts all key-value pairs from a list of table cells.

    Args:
        cells (list): List of BeautifulSoup 'td' elements.
        section (str): Current section being parsed ('Settings' or 'Results').

    Returns:
        list: List of tuples containing (label, value).
    """
    pairs = []
    i = 0
    input_mode = False
    while i < len(cells):
        cell_text = cells[i].get_text(strip=True)
        if cell_text.endswith(':'):
            label = cell_text.rstrip(':').strip()
            if label == 'Inputs':
                input_mode = True
                pairs.append((label, ''))
                i += 1
                continue
            value = ''
            if i + 1 < len(cells):
                next_cell = cells[i + 1]
                value = next_cell.get_text(strip=True)
                j = i + 2
                while j < len(cells) and not cells[j].get_text(strip=True).endswith(':'):
                    value += ' ' + cells[j].get_text(strip=True)
                    j += 1
                i = j - 1
            pairs.append((label, value))
        elif input_mode and section == 'Settings':
            # For inputs, treat each cell as a separate key-value pair
            input_pair = cell_text.split('=', 1)
            if len(input_pair) == 2:
                pairs.append((input_pair[0].strip(), input_pair[1].strip()))
            else:
                pairs.append((f'Input_{i}', cell_text))
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
                headers = ['Open Time', 'Order', 'Symbol', 'Type', 'Volume', 'Price', 'S/L', 'T/P', 'Time', 'State', 'Comment']
                writer.writerow(headers)
                writer.writerows(data)
            elif filename.startswith('deals'):
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
    if data.get('Settings'):
        save_to_csv(data['Settings'], 'settings_data.csv')
    if data.get('Results'):
        save_to_csv(data['Results'], 'results_data.csv')

def save_orders_and_deals_to_csv(data):
    """
    Saves the Orders and Deals data into separate CSV files.

    Args:
        data (dict): Dictionary containing 'Orders' and 'Deals'.
    """
    if data.get('Orders'):
        save_to_csv(data['Orders'], 'orders_data.csv')
    if data.get('Deals'):
        save_to_csv(data['Deals'], 'deals_data.csv')

if __name__ == "__main__":
    html_file_path = r"C:\Users\StdUser\Desktop\MyProjects\Backtesting\logs\AMD_10021629_report.html"

    if not os.path.exists(html_file_path):
        print(f"HTML file '{html_file_path}' does not exist. Please check the path.")
    else:
        data = parse_strategy_tester_report(html_file_path)
        save_settings_and_results_to_csv(data)
        save_orders_and_deals_to_csv(data)

        print("\nExtracted Data:")
        for section in ['Settings', 'Results', 'Orders', 'Deals']:
            print(f"\n--- {section} ---")
            for item in data.get(section, []):
                print(item)