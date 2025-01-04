from bs4 import BeautifulSoup
import csv

def parse_strategy_tester_report(html_file):
    """
    Parses the Strategy Tester Report HTML file and extracts relevant data.

    Args:
        html_file (str): Path to the HTML file.

    Returns:
        dict: Extracted data categorized into 'Results', 'Orders', and 'Deals'.
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
    results_data = []
    orders_data = []
    deals_data = []

    for table_index, table in enumerate(tables, start=1):
        print(f"\n{'='*40}\nParsing Table {table_index}\n{'='*40}")
        rows = table.find_all('tr')

        current_section = None  # To track if we're in 'Orders' or 'Deals'

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
                    current_section = 'Results'
                    continue

            # Skip rows containing images
            if row.find('img'):
                continue

            # Find all table data cells in the row
            cells = row.find_all('td')

            # Skip rows that don't have any cells
            if not cells:
                continue

            if current_section == 'Results':
                # Extract label-value pairs
                label = None
                value = None
                for i, cell in enumerate(cells):
                    cell_text = cell.get_text(strip=True)
                    if cell_text.endswith(':'):
                        label = cell_text.rstrip(':')
                        # Handle colspan by considering the next cell(s)
                        value_cells = cells[i + 1:]
                        value = ' '.join([c.get_text(strip=True) for c in value_cells])
                        break
                if label and value:
                    print(f"{label}: {value}")
                    results_data.append((label, value))

            elif current_section in ['Orders', 'Deals']:
                # Extract data based on the section
                row_data = [cell.get_text(strip=True) for cell in cells]
                print(f"{current_section} Row: {row_data}")
                if current_section == 'Orders':
                    orders_data.append(row_data)
                elif current_section == 'Deals':
                    deals_data.append(row_data)

        print(f"Finished parsing Table {table_index}\n")

    return {
        'Results': results_data,
        'Orders': orders_data,
        'Deals': deals_data
    }

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
            if filename.startswith('results'):
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

if __name__ == "__main__":
    # Replace 'report.html' with the path to your HTML file
    html_file_path = r"C:\Users\StdUser\Desktop\MyProjects\Backtesting\logs\AMD_10008151_output_report.html"  # Replace with your file path

    # Parse the HTML report
    data = parse_strategy_tester_report(html_file_path)

    # Save Results to CSV
    if data.get('Results'):
        save_to_csv(data['Results'], 'results_data.csv')

    # Save Orders to CSV
    if data.get('Orders'):
        save_to_csv(data['Orders'], 'orders_data.csv')

    # Save Deals to CSV
    if data.get('Deals'):
        save_to_csv(data['Deals'], 'deals_data.csv')

    # Optionally, print all extracted data
    print("\nExtracted Data:")
    for section, items in data.items():
        print(f"\n--- {section} ---")
        for item in items:
            print(item)
