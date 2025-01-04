# Database connection parameters
DB_CONFIG = {
    'host': 'ep-morning-tree-a6vcm1gg.us-west-2.retooldb.com',  # e.g., 'localhost'
    'port': 5432,                # Default PostgreSQL port
    'dbname': 'retool',          # Your database name
    'user': 'retool',            # Your database username
    'password': 'gC5bAfzS7yXc',  # Your database password
}

# Python code
import os
import re
import csv
import uuid
import psycopg2
from psycopg2 import sql
from bs4 import BeautifulSoup

def extract_run_id_from_filename(html_file_path):
    file_name = os.path.basename(html_file_path)
    match = re.match(r"^(.*?)_report\.html$", file_name)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Filename '{file_name}' does not match the expected format '<run_id>_report.html'")

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
                    print(f"Skipping header row in {current_section} section.")
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
                print(f"{current_section} Row: {row_data}")
                
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

    print(f"Finished parsing Table {table_index}\n")
    print("================================")
    print(settings_data)
    print("================================")

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

def save_to_csv(data, filename):
    """
    Saves the provided data to a CSV file.
    
    :param data: List of tuples or lists containing the data.
    :param filename: The name of the CSV file to save the data.
    """
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if data and isinstance(data[0], tuple):
                # For Settings and Results
                writer.writerow(['Label', 'Value'])
                writer.writerows(data)
            elif data and isinstance(data[0], list):
                # For Orders and Deals
                writer.writerows(data)
        print(f"Data successfully saved to '{filename}'.")
    except Exception as e:
        print(f"Failed to save data to '{filename}': {e}")

def save_settings_and_results_to_csv(data, run_id):
    if data.get('Settings'):
        save_to_csv(data['Settings'], f'{run_id}_settings_data.csv')
    if data.get('Results'):
        save_to_csv(data['Results'], f'{run_id}_results_data.csv')

def save_orders_and_deals_to_csv(data, run_id):
    if data.get('Orders'):
        save_to_csv(data['Orders'], f'{run_id}_orders_data.csv')
    if data.get('Deals'):
        save_to_csv(data['Deals'], f'{run_id}_deals_data.csv')

def connect_to_db():
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            dbname=DB_CONFIG['dbname'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        print("Database connection established.")
        return conn
    except Exception as e:
        print(f"Failed to connect to the database: {e}")
        return None

def create_or_alter_table(conn, table_name, required_columns):
    """
    Creates or alters a table with the required columns.
    
    :param conn: The database connection object.
    :param table_name: The name of the table to create or alter.
    :param required_columns: A list of (column_name, data_type) tuples for the table.
    """
    try:
        with conn.cursor() as cursor:
            # Create table if it doesn't exist with all required columns
            columns_definition = ',\n    '.join([f"{col} {dtype}" for col, dtype in required_columns])
            create_table_query = sql.SQL("""
                CREATE TABLE IF NOT EXISTS {table} (
                    {columns}
                );
            """).format(
                table=sql.Identifier(table_name),
                columns=sql.SQL(columns_definition)
            )
            cursor.execute(create_table_query)
            print(f"Table '{table_name}' ensured to exist.")

            # Fetch existing columns
            cursor.execute("""
                SELECT LOWER(column_name) 
                FROM information_schema.columns 
                WHERE LOWER(table_name) = LOWER(%s);
            """, (table_name,))
            existing_columns = {row[0] for row in cursor.fetchall()}

            # Iterate over required columns and add any that are missing
            for column_name, data_type in required_columns:
                if column_name.lower() not in existing_columns:
                    alter_table_query = sql.SQL("""
                        ALTER TABLE {table} ADD COLUMN {column} {datatype};
                    """).format(
                        table=sql.Identifier(table_name),
                        column=sql.Identifier(column_name),
                        datatype=sql.SQL(data_type)
                    )
                    cursor.execute(alter_table_query)
                    print(f"Added '{column_name}' column to '{table_name}'.")

            # Ensure 'uuid' is set as PRIMARY KEY if present
            if 'uuid' in [col[0] for col in required_columns]:
                # Check if a primary key already exists
                cursor.execute("""
                    SELECT constraint_name 
                    FROM information_schema.table_constraints 
                    WHERE table_name = %s AND constraint_type = 'PRIMARY KEY';
                """, (table_name,))
                if not cursor.fetchone():
                    # Add primary key constraint on 'uuid'
                    primary_key_query = sql.SQL("""
                        ALTER TABLE {table} ADD PRIMARY KEY (uuid);
                    """).format(
                        table=sql.Identifier(table_name)
                    )
                    cursor.execute(primary_key_query)
                    print(f"Set 'uuid' as PRIMARY KEY for '{table_name}'.")

            conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Failed to create or alter table '{table_name}': {e}")

def create_or_alter_table_run_orders(conn):
    """
    Creates or alters the 'run_orders' table with the required columns.
    
    :param conn: The database connection object.
    """
    required_columns = [
        ('uuid', 'UUID PRIMARY KEY'),
        ('run_id', 'VARCHAR(255)'),
        ('open_time', 'TIMESTAMP'),
        ('run_order', 'VARCHAR(255)'),  # Ensure the use of "order" with quotes
        ('symbol', 'VARCHAR(50)'),
        ('type', 'VARCHAR(50)'),
        ('volume', 'FLOAT'),
        ('price', 'FLOAT'),
        ('s_l', 'FLOAT'),      # Stop Loss
        ('t_p', 'FLOAT'),      # Take Profit
        ('time', 'TIMESTAMP'),
        ('state', 'VARCHAR(50)'),
        ('comment', 'TEXT')
    ]
    
    create_or_alter_table(conn, 'run_orders', required_columns)

def insert_orders_into_db(conn, orders, run_id):
    """
    Inserts orders data into the 'run_orders' table.
    
    :param conn: The database connection object.
    :param orders: List of orders data.
    :param run_id: The run identifier.
    """
    insert_query = """
        INSERT INTO run_orders (uuid, run_id, open_time, run_order, symbol, type, volume, price, s_l, t_p, time, state, comment)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (uuid) DO NOTHING;
    """
    
    try:
        with conn.cursor() as cursor:
            for order in orders:
                if len(order) < 11:
                    print(f"Skipping incomplete order data: {order}")
                    continue
                # Convert data types as necessary
                open_time = convert_to_timestamp(order[0])
                time = convert_to_timestamp(order[8])
                try:
                    volume = float(order[4].replace(',', ''))
                except ValueError:
                    volume = None
                try:
                    price = float(order[5].replace(',', ''))
                except ValueError:
                    price = None
                try:
                    s_l = float(order[6].replace(',', ''))
                except ValueError:
                    s_l = None
                try:
                    t_p = float(order[7].replace(',', ''))
                except ValueError:
                    t_p = None

                cursor.execute(insert_query, (
                    str(uuid.uuid4()),  # uuid
                    run_id,
                    open_time,         # open_time
                    order[1],          # order (ensure consistent quoting here)
                    order[2],          # symbol
                    order[3],          # type
                    volume,            # volume
                    price,             # price
                    s_l,               # s_l
                    t_p,               # t_p
                    time,              # time
                    order[9],          # state
                    order[10]          # comment
                ))
        conn.commit()
        print(f"Inserted {len(orders)} orders into 'run_orders'.")
    except Exception as e:
        conn.rollback()
        print(f"Failed to insert orders into 'run_orders': {e}")

def create_or_alter_table_run_deals(conn):
    """
    Creates or alters the 'run_deals' table with the required columns.
    
    :param conn: The database connection object.
    """
    required_columns = [
        ('uuid', 'UUID PRIMARY KEY'),
        ('run_id', 'VARCHAR(255)'),
        ('time', 'TIMESTAMP'),
        ('deal', 'VARCHAR(255)'),
        ('symbol', 'VARCHAR(50)'),
        ('type', 'VARCHAR(50)'),
        ('direction', 'VARCHAR(10)'),
        ('volume', 'FLOAT'),
        ('price', 'FLOAT'),
        ('run_order', 'VARCHAR(255)'),
        ('commission', 'FLOAT'),
        ('swap', 'FLOAT'),
        ('profit', 'FLOAT'),
        ('balance', 'FLOAT'),
        ('comment', 'TEXT')
    ]
    
    create_or_alter_table(conn, 'run_deals', required_columns)

def convert_to_timestamp(value):
    """
    Converts a string to a PostgreSQL TIMESTAMP format.
    
    :param value: The string representation of the timestamp.
    :return: A string in 'YYYY-MM-DD HH:MM:SS' format or None if conversion fails.
    """
    # Implement appropriate parsing based on your timestamp format
    # Placeholder implementation:
    try:
        # Example: '2023-11-01 12:34:56'
        return value.strip()
    except:
        return None

def insert_deals_into_db(conn, deals, run_id):
    """
    Inserts deals data into the 'run_deals' table.
    
    :param conn: The database connection object.
    :param deals: List of deals data.
    :param run_id: The run identifier.
    """
    insert_query = """
        INSERT INTO run_deals (uuid, run_id, time, deal, symbol, type, direction, volume, price, run_order, commission, swap, profit, balance, comment)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (uuid) DO NOTHING;
    """
    
    try:
        with conn.cursor() as cursor:
            for deal in deals:
                if len(deal) < 13:
                    print(f"Skipping incomplete deal data: {deal}")
                    continue
                # Convert data types as necessary
                time = convert_to_timestamp(deal[0])
                try:
                    volume = float(deal[5].replace(',', ''))
                except ValueError:
                    volume = None
                try:
                    price = float(deal[6].replace(',', ''))
                except ValueError:
                    price = None
                try:
                    commission = float(deal[8].replace(',', ''))
                except ValueError:
                    commission = None
                try:
                    swap = float(deal[9].replace(',', ''))
                except ValueError:
                    swap = None
                try:
                    profit = float(deal[10].replace(',', ''))
                except ValueError:
                    profit = None
                try:
                    balance = float(deal[11].replace(',', ''))
                except ValueError:
                    balance = None

                cursor.execute(insert_query, (
                    str(uuid.uuid4()),  # uuid
                    run_id,
                    time,              # time
                    deal[1],           # deal
                    deal[2],           # symbol
                    deal[3],           # type
                    deal[4],           # direction
                    volume,            # volume
                    price,             # price
                    deal[7],           # order
                    commission,        # commission
                    swap,              # swap
                    profit,            # profit
                    balance,           # balance
                    deal[12]           # comment
                ))
        conn.commit()
        print(f"Inserted {len(deals)} deals into 'run_deals'.")
    except Exception as e:
        conn.rollback()
        print(f"Failed to insert deals into 'run_deals': {e}")

def insert_data_into_db(conn, data, run_id):
    if data.get('Orders'):
        insert_orders_into_db(conn, data['Orders'], run_id)
    if data.get('Deals'):
        insert_deals_into_db(conn, data['Deals'], run_id)

def insert_run_performance_into_db(conn, settings, results, run_id):
    """
    Inserts data into the 'run_performance' table with fixed columns.
    
    :param conn: The database connection object.
    :param settings: List of settings data as (label, value) tuples.
    :param results: List of results data as (label, value) tuples.
    :param run_id: The run identifier.
    """
    # Define fixed columns and their mapping from labels
    fixed_columns = {
        'Expert': 'expert',
        'Symbol': 'symbol',
        'Period': 'period',
        'Inputs': 'inputs',
        'Company': 'company',
        'Currency': 'currency',
        'Initial Deposit': 'initial_deposit',
        'Leverage': 'leverage',
        'History Quality': 'history_quality',
        'Bars': 'bars',
        'Ticks': 'ticks',
        'Symbols': 'symbols',
        'Total Net Profit': 'total_net_profit',
        'Balance Drawdown Absolute': 'balance_drawdown_absolute',
        'Equity Drawdown Absolute': 'equity_drawdown_absolute',
        'Gross Profit': 'gross_profit',
        'Balance Drawdown Maximal value': 'balance_drawdown_maximal_value',
        'Balance Drawdown Maximal percentage': 'balance_drawdown_maximal_percentage',
        'Equity Drawdown Maximal value': 'equity_drawdown_maximal_value',
        'Equity Drawdown Maximal percentage': 'equity_drawdown_maximal_percentage',
        'Gross Loss': 'gross_loss',
        'Balance Drawdown Relative percentage': 'balance_drawdown_relative_percentage',
        'Balance Drawdown Relative amount': 'balance_drawdown_relative_amount',
        'Equity Drawdown Relative percentage': 'equity_drawdown_relative_percentage',
        'Equity Drawdown Relative amount': 'equity_drawdown_relative_amount',
        'Profit Factor': 'profit_factor',
        'Expected Payoff': 'expected_payoff',
        'Margin Level': 'margin_level',
        'Recovery Factor': 'recovery_factor',
        'Sharpe Ratio': 'sharpe_ratio',
        'Z-Score value': 'z_score_value',
        'Z-Score percentage': 'z_score_percentage',
        'AHPR value': 'ahpr_value',
        'AHPR percentage': 'ahpr_percentage',
        'LR Correlation': 'lr_correlation',
        'OnTester result': 'ontester_result',
        'GHPR value': 'ghpr_value',
        'GHPR percentage': 'ghpr_percentage',
        'LR Standard Error': 'lr_standard_error',
        'Total Trades': 'total_trades',
        'Short Trades won': 'short_trades_won',
        'Short Trades won %': 'short_trades_won_percentage',
        'Long Trades won': 'long_trades_won',
        'Long Trades won %': 'long_trades_won_percentage',
        'Total Deals': 'total_deals',
        'Profit Trades count': 'profit_trades_count',
        'Profit Trades % of total': 'profit_trades_percentage',
        'Loss Trades count': 'loss_trades_count',
        'Loss Trades % of total': 'loss_trades_percentage',
        'Largest profit trade': 'largest_profit_trade',
        'Largest loss trade': 'largest_loss_trade',
        'Average profit trade': 'average_profit_trade',
        'Average loss trade': 'average_loss_trade',
        'Maximum consecutive wins count': 'maximum_consecutive_wins_count',
        'Maximum consecutive wins value': 'maximum_consecutive_wins_value',
        'Maximum consecutive losses count': 'maximum_consecutive_losses_count',
        'Maximum consecutive losses value': 'maximum_consecutive_losses_value',
        'Maximal consecutive profit value': 'maximal_consecutive_profit_value',
        'Maximal consecutive profit count': 'maximal_consecutive_profit_count',
        'Maximal consecutive loss value': 'maximal_consecutive_loss_value',
        'Maximal consecutive loss count': 'maximal_consecutive_loss_count',
        'Average consecutive wins': 'average_consecutive_wins',
        'Average consecutive losses': 'average_consecutive_losses',
        'Correlation (Profits,MFE)': 'correlation_profits_mfe',
        'Correlation (Profits,MAE)': 'correlation_profits_mae',
        'Correlation (MFE,MAE)': 'correlation_mfe_mae',
        'Minimal position holding time': 'minimal_position_holding_time',
        'Maximal position holding time': 'maximal_position_holding_time',
        'Average position holding time': 'average_position_holding_time'
    }

    # Initialize a dictionary to hold column values
    performance_data = {
        'uuid': str(uuid.uuid4()),
        'run_id': run_id,
        'comments': ''  # Adding the comments field
    }

    # Map settings and results to fixed columns
    for label, value in settings + results:
        if label in fixed_columns:
            column = fixed_columns[label]
            # Convert value to appropriate data type
            converted_value = convert_value(column, value)
            performance_data[column] = converted_value
        else:
            print(f"Warning: Label '{label}' not recognized for run_performance table.")

    # Define the fixed columns and their order
    fixed_columns_order = [
        'expert', 'symbol', 'period', 'inputs', 'company', 'currency',
        'initial_deposit', 'leverage', 'history_quality', 'bars',
        'ticks', 'symbols', 'total_net_profit', 'balance_drawdown_absolute',
        'equity_drawdown_absolute', 'gross_profit', 'balance_drawdown_maximal_value',
        'balance_drawdown_maximal_percentage', 'equity_drawdown_maximal_value',
        'equity_drawdown_maximal_percentage', 'gross_loss',
        'balance_drawdown_relative_percentage', 'balance_drawdown_relative_amount',
        'equity_drawdown_relative_percentage', 'equity_drawdown_relative_amount',
        'profit_factor', 'expected_payoff', 'margin_level', 'recovery_factor',
        'sharpe_ratio', 'z_score_value', 'z_score_percentage', 'ahpr_value',
        'ahpr_percentage', 'lr_correlation', 'ontester_result', 'ghpr_value',
        'ghpr_percentage', 'lr_standard_error', 'total_trades', 'short_trades_won',
        'short_trades_won_percentage', 'long_trades_won', 'long_trades_won_percentage',
        'total_deals', 'profit_trades_count', 'profit_trades_percentage',
        'loss_trades_count', 'loss_trades_percentage', 'largest_profit_trade',
        'largest_loss_trade', 'average_profit_trade', 'average_loss_trade',
        'maximum_consecutive_wins_count', 'maximum_consecutive_wins_value',
        'maximum_consecutive_losses_count', 'maximum_consecutive_losses_value',
        'maximal_consecutive_profit_value', 'maximal_consecutive_profit_count',
        'maximal_consecutive_loss_value', 'maximal_consecutive_loss_count',
        'average_consecutive_wins', 'average_consecutive_losses',
        'correlation_profits_mfe', 'correlation_profits_mae', 'correlation_mfe_mae',
        'minimal_position_holding_time', 'maximal_position_holding_time',
        'average_position_holding_time'
    ]

    # Prepare the INSERT statement with fixed columns
    columns = ['uuid', 'run_id', 'comments'] + fixed_columns_order
    placeholders = ', '.join(['%s'] * len(columns))
    insert_query = sql.SQL("""
        INSERT INTO run_performance ({fields})
        VALUES ({placeholders})
        ON CONFLICT (uuid) DO NOTHING;
    """).format(
        fields=sql.SQL(', ').join(map(sql.Identifier, columns)),
        placeholders=sql.SQL(placeholders)
    )

    # Prepare the values in the same order as columns
    values = [performance_data.get(col, None) for col in columns]

    try:
        with conn.cursor() as cursor:
            cursor.execute(insert_query, values)
            conn.commit()
            print(f"Inserted run_id '{run_id}' into 'run_performance'.")
    except psycopg2.IntegrityError as e:
        conn.rollback()
        print(f"Integrity Error: {e}")
        print(f"run_id '{run_id}' may already exist in 'run_performance'.")
    except Exception as e:
        conn.rollback()
        print(f"Failed to insert into 'run_performance': {e}")

def convert_value(column, value):
    """
    Converts the string value to the appropriate data type based on the column.

    :param column: The column name.
    :param value: The string value to convert.
    :return: The value converted to the appropriate data type.
    """
    # Define data type mapping based on column names
    int_columns = [
        'bars', 'ticks', 'symbols', 'ontester_result', 'total_trades',
        'short_trades_won', 'long_trades_won', 'total_deals',
        'profit_trades_count', 'loss_trades_count',
        'maximum_consecutive_wins_count', 'maximum_consecutive_losses_count',
        'maximal_consecutive_profit_count', 'maximal_consecutive_loss_count',
        'average_consecutive_wins', 'average_consecutive_losses'
    ]

    float_columns = [
        'initial_deposit', 'history_quality', 'total_net_profit',
        'balance_drawdown_absolute', 'equity_drawdown_absolute',
        'gross_profit', 'balance_drawdown_maximal_value',
        'balance_drawdown_maximal_percentage', 'equity_drawdown_maximal_value',
        'equity_drawdown_maximal_percentage', 'gross_loss',
        'balance_drawdown_relative_percentage', 'balance_drawdown_relative_amount',
        'equity_drawdown_relative_percentage', 'equity_drawdown_relative_amount',
        'profit_factor', 'expected_payoff', 'margin_level',
        'recovery_factor', 'sharpe_ratio', 'z_score_value',
        'z_score_percentage', 'ahpr_value', 'ahpr_percentage',
        'lr_correlation', 'ghpr_value', 'ghpr_percentage',
        'lr_standard_error', 'profit_trades_percentage', 'loss_trades_percentage',
        'largest_profit_trade', 'largest_loss_trade', 'average_profit_trade',
        'average_loss_trade', 'maximum_consecutive_wins_value',
        'maximum_consecutive_losses_value', 'maximal_consecutive_profit_value',
        'maximal_consecutive_loss_value', 'correlation_profits_mfe',
        'correlation_profits_mae', 'correlation_mfe_mae'
    ]

    bool_columns = ['inputs']  # Since Inputs contains multiple settings, we'll keep it as text

    time_columns = [
        'minimal_position_holding_time', 'maximal_position_holding_time',
        'average_position_holding_time'
    ]

    varchar_columns = [
        'expert', 'symbol', 'period', 'leverage', 'company', 'currency', 'inputs'
    ]

    if column in int_columns:
        try:
            return int(value.replace(',', '').strip())
        except ValueError:
            return None
    elif column in float_columns:
        # Check if value is a string before trying to replace
        if isinstance(value, str):
            value = value.replace('%', '').replace(',', '').strip()
        try:
            return float(value)
        except ValueError:
            return None
    elif column in bool_columns:
        # Keep Inputs as text
        return value
    elif column in time_columns:
        # Convert to HH:MM:SS format
        if isinstance(value, str):
            try:
                parts = value.strip().split(':')
                if len(parts) == 3:
                    return f"{int(parts[0]):02}:{int(parts[1]):02}:{int(parts[2]):02}"
                else:
                    return None
            except:
                return None
    else:
        # Default to string
        return str(value) if isinstance(value, (str, float)) else None

def create_or_alter_table_run_performance(conn):
    """
    Creates or alters the 'run_performance' table with fixed columns using lowercase column names.
    
    :param conn: The database connection object.
    """
    required_columns = [
        ('uuid', 'UUID PRIMARY KEY'),
        ('run_id', 'VARCHAR(255)'),
        ('expert', 'VARCHAR(255)'),  # Lowercase here
        ('symbol', 'VARCHAR(10)'),
        ('period', 'VARCHAR(255)'),
        ('inputs', 'TEXT'),
        ('company', 'VARCHAR(255)'),
        ('currency', 'VARCHAR(10)'),
        ('initial_deposit', 'DECIMAL(10, 2)'),
        ('leverage', 'VARCHAR(10)'),
        ('history_quality', 'DECIMAL(5, 2)'),
        ('bars', 'INTEGER'),
        ('ticks', 'BIGINT'),
        ('symbols', 'INTEGER'),
        ('total_net_profit', 'DECIMAL(10, 2)'),
        ('balance_drawdown_absolute', 'DECIMAL(10, 2)'),
        ('equity_drawdown_absolute', 'DECIMAL(10, 2)'),
        ('gross_profit', 'DECIMAL(10, 2)'),
        ('balance_drawdown_maximal_value', 'DECIMAL(10, 2)'),
        ('balance_drawdown_maximal_percentage', 'DECIMAL(5, 2)'),
        ('equity_drawdown_maximal_value', 'DECIMAL(10, 2)'),
        ('equity_drawdown_maximal_percentage', 'DECIMAL(5, 2)'),
        ('gross_loss', 'DECIMAL(10, 2)'),
        ('balance_drawdown_relative_percentage', 'DECIMAL(5, 2)'),
        ('balance_drawdown_relative_amount', 'DECIMAL(10, 2)'),
        ('equity_drawdown_relative_percentage', 'DECIMAL(5, 2)'),
        ('equity_drawdown_relative_amount', 'DECIMAL(10, 2)'),
        ('profit_factor', 'DECIMAL(5, 2)'),
        ('expected_payoff', 'DECIMAL(5, 2)'),
        ('margin_level', 'DECIMAL(5, 2)'),
        ('recovery_factor', 'DECIMAL(5, 2)'),
        ('sharpe_ratio', 'DECIMAL(5, 2)'),
        ('z_score_value', 'DECIMAL(5, 2)'),
        ('z_score_percentage', 'DECIMAL(5, 2)'),
        ('ahpr_value', 'DECIMAL(5, 4)'),
        ('ahpr_percentage', 'DECIMAL(5, 2)'),
        ('lr_correlation', 'DECIMAL(5, 2)'),
        ('ontester_result', 'INTEGER'),
        ('ghpr_value', 'DECIMAL(5, 4)'),
        ('ghpr_percentage', 'DECIMAL(5, 2)'),
        ('lr_standard_error', 'DECIMAL(5, 2)'),
        ('total_trades', 'INTEGER'),
        ('short_trades_won', 'INTEGER'),
        ('short_trades_won_percentage', 'DECIMAL(5, 2)'),
        ('long_trades_won', 'INTEGER'),
        ('long_trades_won_percentage', 'DECIMAL(5, 2)'),
        ('total_deals', 'INTEGER'),
        ('profit_trades_count', 'INTEGER'),
        ('profit_trades_percentage', 'DECIMAL(5, 2)'),
        ('loss_trades_count', 'INTEGER'),
        ('loss_trades_percentage', 'DECIMAL(5, 2)'),
        ('largest_profit_trade', 'DECIMAL(10, 2)'),
        ('largest_loss_trade', 'DECIMAL(10, 2)'),
        ('average_profit_trade', 'DECIMAL(10, 2)'),
        ('average_loss_trade', 'DECIMAL(10, 2)'),
        ('maximum_consecutive_wins_count', 'INTEGER'),
        ('maximum_consecutive_wins_value', 'DECIMAL(10, 2)'),
        ('maximum_consecutive_losses_count', 'INTEGER'),
        ('maximum_consecutive_losses_value', 'DECIMAL(10, 2)'),
        ('maximal_consecutive_profit_value', 'DECIMAL(10, 2)'),
        ('maximal_consecutive_profit_count', 'INTEGER'),
        ('maximal_consecutive_loss_value', 'DECIMAL(10, 2)'),
        ('maximal_consecutive_loss_count', 'INTEGER'),
        ('average_consecutive_wins', 'INTEGER'),
        ('average_consecutive_losses', 'INTEGER'),
        ('correlation_profits_mfe', 'DECIMAL(5, 2)'),
        ('correlation_profits_mae', 'DECIMAL(5, 2)'),
        ('correlation_mfe_mae', 'DECIMAL(7, 4)'),
        ('minimal_position_holding_time', 'TIME'),
        ('maximal_position_holding_time', 'TIME'),
        ('average_position_holding_time', 'TIME'),
        ('comments', 'TEXT')
    ]

    create_or_alter_table(conn, 'run_performance', required_columns)


if __name__ == "__main__":
    # Replace with your actual HTML file path or modify to accept as an argument
    html_file_path = r"C:\Users\StdUser\Desktop\MyProjects\Backtesting\logs\AMD_10021629_report.html"  # Example file path
    # html_file_path = r"C:/Users/StdUser/Desktop/MyProjects/Backtesting/logs/AMD_10014704_report.html"  # Example file path

    # Check if the HTML file exists
    if not os.path.exists(html_file_path):
        print(f"HTML file '{html_file_path}' does not exist. Please check the path.")
    else:
        # Extract the run_id from the HTML file name
        try:
            run_id = extract_run_id_from_filename(html_file_path)
        except ValueError as e:
            print(e)
            exit(1)

        # Parse the HTML report
        data = parse_strategy_tester_report(html_file_path)

        # Save Settings and Results to CSV
        save_settings_and_results_to_csv(data, run_id)

        # Save Orders and Deals to CSV
        save_orders_and_deals_to_csv(data, run_id)

        # Optionally, print all extracted data for verification
        print("\nExtracted Data:")

        # Print Settings
        print("\n--- Settings ---")
        for label, value in data.get('Settings', []):
            print(f"{label}: {value}")

        # # Print Results
        # print("\n--- Results ---")
        # for label, value in data.get('Results', []):
        #     print(f"{label}: {value}")

        # # Print Orders
        # print("\n--- Orders ---")
        # for order in data.get('Orders', []):
        #     print(order)

        # # Print Deals
        # print("\n--- Deals ---")
        # for deal in data.get('Deals', []):
        #     print(deal)

        # Connect to the database
        conn = connect_to_db()
        if conn:
            try:
                # Create or alter run_performance table with fixed columns
                create_or_alter_table_run_performance(conn)

                # Insert into run_performance
                insert_run_performance_into_db(conn, data.get('Settings', []), data.get('Results', []), run_id)

                # Create or alter other tables
                create_or_alter_table_run_orders(conn)
                create_or_alter_table_run_deals(conn)

                # Insert data into the database
                insert_data_into_db(conn, data, run_id)

                # Optionally, insert Settings and Results into the database
                # insert_settings_and_results_into_db(conn, data, run_id)

            finally:
                # Close the database connection
                conn.close()
                print("Database connection closed.")
