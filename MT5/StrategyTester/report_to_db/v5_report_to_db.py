from bs4 import BeautifulSoup
import csv
import os

def parse_strategy_tester_report(html_file):
    try:
        with open(html_file, 'r', encoding='utf-16', errors='replace') as file:
            soup = BeautifulSoup(file, 'html.parser')
    except Exception as e:
        print(f"Failed to read the file with UTF-16 encoding: {e}")
        return {}

    settings_data = []
    results_data = []
    orders_data = []
    deals_data = []

    # Find the first table (assumed to contain settings and results)
    main_table = soup.find('table')
    if main_table:
        rows = main_table.find_all('tr')
        current_section = 'Settings'
        inputs = []
        input_mode = False

        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 2:
                label = cells[0].get_text(strip=True).rstrip(':')
                value = cells[1].get_text(strip=True)

                if label == 'Inputs':
                    input_mode = True
                    inputs.append(value)
                elif input_mode and not label:
                    # Continuation of inputs
                    inputs.append(value)
                else:
                    if input_mode:
                        # End of inputs section
                        settings_data.append(('Inputs', '; '.join(inputs)))
                        input_mode = False
                        inputs = []

                    if label == 'Symbol':
                        current_section = 'Results'
                    
                    if current_section == 'Settings':
                        settings_data.append((label, value))
                    else:
                        results_data.append((label, value))

        if inputs:
            # In case inputs are at the end
            settings_data.append(('Inputs', '; '.join(inputs)))

    # Parse Orders and Deals tables (if present)
    tables = soup.find_all('table')
    if len(tables) > 1:
        for table in tables[1:]:
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            if 'Order' in headers:
                orders_data = [
                    [td.get_text(strip=True) for td in row.find_all('td')]
                    for row in table.find_all('tr')[1:]
                ]
            elif 'Deal' in headers:
                deals_data = [
                    [td.get_text(strip=True) for td in row.find_all('td')]
                    for row in table.find_all('tr')[1:]
                ]

    return {
        'Settings': settings_data,
        'Results': results_data,
        'Orders': orders_data,
        'Deals': deals_data
    }

def save_to_csv(data, filename):
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if filename.startswith('settings') or filename.startswith('results'):
                writer.writerow(['Label', 'Value'])
                for label, value in data:
                    if label == 'Inputs':
                        writer.writerow([label, ''])
                        for input_pair in value.split('; '):
                            writer.writerow(['', input_pair])
                    else:
                        writer.writerow([label, value])
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
    if data.get('Settings'):
        save_to_csv(data['Settings'], 'settings_data.csv')
    if data.get('Results'):
        save_to_csv(data['Results'], 'results_data.csv')

def save_orders_and_deals_to_csv(data):
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
        for section in ['Settings', 'Results']:
            print(f"\n--- {section} ---")
            for label, value in data.get(section, []):
                if label == 'Inputs':
                    print(f"{label}:")
                    for input_pair in value.split('; '):
                        print(f"  {input_pair}")
                else:
                    print(f"{label}: {value}")

        for section in ['Orders', 'Deals']:
            print(f"\n--- {section} ---")
            for item in data.get(section, []):
                print(item)