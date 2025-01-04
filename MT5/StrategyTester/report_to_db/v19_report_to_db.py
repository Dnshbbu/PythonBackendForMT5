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

def split_specific_fields(label, value, fields_to_split):
    """
    Splits specific labels and values containing parentheses into separate key-value pairs.

    Args:
        label (str): The original label.
        value (str): The original value.
        fields_to_split (set): Set of labels that require splitting.

    Returns:
        list: List of tuples with split labels and values.
    """
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
    Sanitizes the label to create a valid SQL column name.
    Converts to lowercase, replaces spaces and special characters with underscores.

    Args:
        label (str): The original label.

    Returns:
        str: Sanitized column name.
    """
    # Convert to lowercase
    column = label.lower()
    # Replace spaces and slashes with underscores
    column = re.sub(r'[ /]+', '_', column)
    # Remove any characters that are not alphanumeric or underscores
    column = re.sub(r'[^a-z0-9_]', '', column)
    return column

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
            if 'settings' in filename:
                # Optionally, write headers
                # writer.writerow(['Label', 'Value'])
                writer.writerows(data)
            elif 'orders' in filename:
                # Optionally, write headers
                # headers = ['Open Time', 'Order', 'Symbol', 'Type', 'Volume', 'Price', 'S/L', 'T/P', 'Time', 'State', 'Comment']
                # writer.writerow(headers)
                writer.writerows(data)
            elif 'deals' in filename:
                # Optionally, write headers
                # headers = ['Time', 'Deal', 'Symbol', 'Type', 'Direction', 'Volume', 'Price', 'Order', 'Commission', 'Swap', 'Profit', 'Balance', 'Comment']
                # writer.writerow(headers)
                writer.writerows(data)
        print(f"Data successfully saved to '{filename}'.")
    except Exception as e:
        print(f"Failed to save data to '{filename}': {e}")

def save_settings_and_results_to_csv(data, run_id):
    """
    Saves the Settings and Results data into separate CSV files with the run_id as prefix.

    Args:
        data (dict): Dictionary containing 'Settings' and 'Results'.
        run_id (str): The run ID to prefix the CSV filenames.
    """
    if data.get('Settings'):
        save_to_csv(data['Settings'], f'{run_id}_settings_data.csv')
    if data.get('Results'):
        save_to_csv(data['Results'], f'{run_id}_results_data.csv')

def save_orders_and_deals_to_csv(data, run_id):
    """
    Saves the Orders and Deals data into separate CSV files with the run_id as prefix.

    Args:
        data (dict): Dictionary containing 'Orders' and 'Deals'.
        run_id (str): The run ID to prefix the CSV filenames.
    """
    if data.get('Orders'):
        save_to_csv(data['Orders'], f'{run_id}_orders_data.csv')
    if data.get('Deals'):
        save_to_csv(data['Deals'], f'{run_id}_deals_data.csv')

def connect_to_db():
    """
    Establishes a connection to the PostgreSQL database using the provided DB_CONFIG.

    Returns:
        connection: A psycopg2 connection object.
    """
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
    Creates the table if it does not exist and adds any missing required columns.

    Args:
        conn: A psycopg2 connection object.
        table_name (str): Name of the table to create or alter.
        required_columns (list of tuples): List of (column_name, data_type) tuples.
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

            # Retrieve existing columns in the table
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s;
            """, (table_name,))
            existing_columns = {row[0] for row in cursor.fetchall()}

            # Iterate over required columns and add any that are missing
            for column_name, data_type in required_columns:
                if column_name not in existing_columns:
                    alter_table_query = sql.SQL("""
                        ALTER TABLE {table} ADD COLUMN {column} {datatype};
                    """).format(
                        table=sql.Identifier(table_name),
                        column=sql.Identifier(column_name),
                        datatype=sql.SQL(data_type)
                    )
                    cursor.execute(alter_table_query)
                    print(f"Added '{column_name}' column to '{table_name}'.")

            # Ensure 'uuid' is set as PRIMARY KEY
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
    Creates or alters the run_orders table to ensure it has all necessary columns.
    """
    required_columns = [
        ('uuid', 'UUID PRIMARY KEY'),
        ('run_id', 'VARCHAR(255)'),
        ('open_time', 'VARCHAR(255)'),
        ('order_id', 'VARCHAR(255)'),  # Changed to VARCHAR if necessary
        ('symbol', 'VARCHAR(255)'),
        ('type', 'VARCHAR(255)'),
        ('volume', 'VARCHAR(255)'),
        ('price', 'VARCHAR(255)'),
        ('sl', 'VARCHAR(255)'),
        ('tp', 'VARCHAR(255)'),
        ('time', 'VARCHAR(255)'),
        ('state', 'VARCHAR(255)'),
        ('comment', 'VARCHAR(255)')
    ]
    create_or_alter_table(conn, 'run_orders', required_columns)

def create_or_alter_table_run_deals(conn):
    """
    Creates or alters the run_deals table to ensure it has all necessary columns.
    """
    required_columns = [
        ('uuid', 'UUID PRIMARY KEY'),
        ('run_id', 'VARCHAR(255)'),
        ('time', 'VARCHAR(255)'),
        ('deal_id', 'VARCHAR(255)'),  # Changed to VARCHAR if necessary
        ('symbol', 'VARCHAR(255)'),
        ('type', 'VARCHAR(255)'),
        ('direction', 'VARCHAR(255)'),
        ('volume', 'VARCHAR(255)'),
        ('price', 'VARCHAR(255)'),
        ('order_id', 'VARCHAR(255)'),
        ('commission', 'VARCHAR(255)'),
        ('swap', 'VARCHAR(255)'),
        ('profit', 'VARCHAR(255)'),
        ('balance', 'VARCHAR(255)'),
        ('comment', 'VARCHAR(255)')
    ]
    create_or_alter_table(conn, 'run_deals', required_columns)

def create_or_alter_table_run_performance(conn, labels):
    """
    Creates or alters the run_performance table to ensure it has all necessary columns based on labels.

    Args:
        conn: A psycopg2 connection object.
        labels (set): A set of labels from Settings and Results.
    """
    # Start with 'uuid' and 'run_id'
    required_columns = [
        ('uuid', 'UUID PRIMARY KEY'),
        ('run_id', 'VARCHAR(2555555)')
    ]

    # Add columns based on labels
    for label in labels:
        column_name = sanitize_column_name(label)
        # Assuming all dynamic columns are VARCHAR; adjust as needed
        required_columns.append((column_name, 'VARCHAR(2555555)'))

    create_or_alter_table(conn, 'run_performance', required_columns)

def sanitize_column_name(label):
    """
    Sanitizes the label to create a valid SQL column name.
    Converts to lowercase, replaces spaces and special characters with underscores.

    Args:
        label (str): The original label.

    Returns:
        str: Sanitized column name.
    """
    # Convert to lowercase
    column = label.lower()
    # Replace spaces and slashes with underscores
    column = re.sub(r'[ /]+', '_', column)
    # Remove any characters that are not alphanumeric or underscores
    column = re.sub(r'[^a-z0-9_]', '', column)
    return column

def insert_run_performance_into_db(conn, settings, results, run_id):
    """
    Inserts a new run_performance record into the database.

    Args:
        conn: A psycopg2 connection object.
        settings (list of tuples): List containing ('Label', 'Value') for settings.
        results (list of tuples): List containing ('Label', 'Value') for results.
        run_id (str): The run ID to associate with the record.
    """
    # Combine settings and results
    combined_data = settings + results

    # Create a dictionary with sanitized column names
    performance_data = {}
    for label, value in combined_data:
        column_name = sanitize_column_name(label)
        performance_data[column_name] = value

    # Add 'uuid' and 'run_id'
    performance_data['uuid'] = str(uuid.uuid4())
    performance_data['run_id'] = run_id

    # Prepare the INSERT statement dynamically
    columns = ', '.join(performance_data.keys())
    placeholders = ', '.join(['%s'] * len(performance_data))
    insert_query = f"INSERT INTO run_performance ({columns}) VALUES ({placeholders}) ON CONFLICT (run_id) DO NOTHING"

    # Prepare the values in the same order as columns
    values = list(performance_data.values())

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

def insert_orders_into_db(conn, orders, run_id):
    """
    Inserts Orders data into the run_orders table.

    Args:
        conn: A psycopg2 connection object.
        orders (list): List of Orders data.
        run_id (str): The run ID to associate with each row.
    """
    insert_query = """
    INSERT INTO run_orders (
        uuid, run_id, open_time, order_id, symbol, type, volume, price, sl, tp, time, state, comment
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    """
    try:
        with conn.cursor() as cursor:
            records = []
            for order in orders:
                # Data validation and conversion
                try:
                    # Convert 'order_id' to integer if necessary
                    # If 'order_id' is supposed to be VARCHAR, skip conversion
                    # order_id = int(order[1])  # Uncomment if 'order_id' is INTEGER
                    order_id = order[1]  # Use as string

                    # Similarly, convert other fields if needed
                    # For example, 'volume' might need to be a float
                    # volume = float(order[4].replace('-', '').replace(',', '.'))  # Example conversion

                    record = (
                        str(uuid.uuid4()),  # uuid
                        run_id,             # run_id
                        order[0],           # Open Time
                        order_id,           # Order
                        order[2],           # Symbol
                        order[3],           # Type
                        order[4],           # Volume
                        order[5],           # Price
                        order[6] if order[6] else None,  # S/L
                        order[7] if order[7] else None,  # T/P
                        order[8],           # Time
                        order[9],           # State
                        order[10]           # Comment
                    )
                    records.append(record)
                except Exception as e:
                    print(f"Skipping invalid order row {order}: {e}")
                    continue  # Skip invalid rows
            cursor.executemany(insert_query, records)
            conn.commit()
            print(f"Inserted {len(records)} records into 'run_orders'.")
    except Exception as e:
        conn.rollback()
        print(f"Failed to insert records into 'run_orders': {e}")

def insert_deals_into_db(conn, deals, run_id):
    """
    Inserts Deals data into the run_deals table.

    Args:
        conn: A psycopg2 connection object.
        deals (list): List of Deals data.
        run_id (str): The run ID to associate with each row.
    """
    insert_query = """
    INSERT INTO run_deals (
        uuid, run_id, time, deal_id, symbol, type, direction, volume, price, order_id,
        commission, swap, profit, balance, comment
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    """
    try:
        with conn.cursor() as cursor:
            records = []
            for deal in deals:
                # Data validation and conversion
                try:
                    # Convert 'deal_id' to integer if necessary
                    # If 'deal_id' is supposed to be VARCHAR, skip conversion
                    # deal_id = int(deal[1])  # Uncomment if 'deal_id' is INTEGER
                    deal_id = deal[1]  # Use as string

                    # Similarly, convert other fields if needed
                    # For example, 'volume' might need to be a float
                    # volume = float(deal[5].replace('-', '').replace(',', '.'))  # Example conversion

                    record = (
                        str(uuid.uuid4()),  # uuid
                        run_id,             # run_id
                        deal[0],            # Time
                        deal_id,            # Deal
                        deal[2],            # Symbol
                        deal[3],            # Type
                        deal[4],            # Direction
                        deal[5],            # Volume
                        deal[6],            # Price
                        deal[7],            # Order
                        deal[8] if deal[8] else None,  # Commission
                        deal[9] if deal[9] else None,  # Swap
                        deal[10] if deal[10] else None, # Profit
                        deal[11] if deal[11] else None, # Balance
                        deal[12]            # Comment
                    )
                    records.append(record)
                except Exception as e:
                    print(f"Skipping invalid deal row {deal}: {e}")
                    continue  # Skip invalid rows
            cursor.executemany(insert_query, records)
            conn.commit()
            print(f"Inserted {len(records)} records into 'run_deals'.")
    except Exception as e:
        conn.rollback()
        print(f"Failed to insert records into 'run_deals': {e}")

def insert_data_into_db(conn, data, run_id):
    """
    Inserts Orders and Deals data into the database.

    Args:
        conn: A psycopg2 connection object.
        data (dict): Dictionary containing 'Orders' and 'Deals'.
        run_id (str): The run ID to associate with each row.
    """
    if data.get('Orders'):
        insert_orders_into_db(conn, data['Orders'], run_id)
    if data.get('Deals'):
        insert_deals_into_db(conn, data['Deals'], run_id)

def insert_settings_and_results_into_db(conn, data, run_id):
    """
    Inserts Settings and Results data into the run_performance table.

    Args:
        conn: A psycopg2 connection object.
        data (dict): Dictionary containing 'Settings' and 'Results'.
        run_id (str): The run ID to associate with each row.
    """
    # This function is now integrated into insert_run_performance_into_db
    pass  # Placeholder for future implementation if needed

def insert_run_performance_into_db(conn, settings, results, run_id):
    """
    Inserts a new run_performance record into the database.

    Args:
        conn: A psycopg2 connection object.
        settings (list of tuples): List containing ('Label', 'Value') for settings.
        results (list of tuples): List containing ('Label', 'Value') for results.
        run_id (str): The run ID to associate with the record.
    """
    # Combine settings and results
    combined_data = settings + results

    # Create a dictionary with sanitized column names
    performance_data = {}
    for label, value in combined_data:
        column_name = sanitize_column_name(label)
        performance_data[column_name] = value

    # Add 'uuid' and 'run_id'
    performance_data['uuid'] = str(uuid.uuid4())
    performance_data['run_id'] = run_id

    # Prepare the INSERT statement dynamically
    columns = ', '.join(performance_data.keys())
    placeholders = ', '.join(['%s'] * len(performance_data))
    insert_query = f"INSERT INTO run_performance ({columns}) VALUES ({placeholders}) ON CONFLICT (run_id) DO NOTHING"

    # Prepare the values in the same order as columns
    values = list(performance_data.values())

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

def create_or_alter_table_run_performance(conn, labels):
    """
    Creates or alters the run_performance table to ensure it has all necessary columns based on labels.

    Args:
        conn: A psycopg2 connection object.
        labels (set): A set of labels from Settings and Results.
    """
    # Start with 'uuid' and 'run_id'
    required_columns = [
        ('uuid', 'UUID PRIMARY KEY'),
        ('run_id', 'VARCHAR(255)')
    ]

    # Add columns based on labels
    for label in labels:
        column_name = sanitize_column_name(label)
        # Assuming all dynamic columns are VARCHAR; adjust as needed
        required_columns.append((column_name, 'VARCHAR(255)'))

    create_or_alter_table(conn, 'run_performance', required_columns)

if __name__ == "__main__":
    # Replace with your actual HTML file path or modify to accept as an argument
    html_file_path = r"C:\Users\StdUser\Desktop\MyProjects\Backtesting\logs\AMD_10021629_report.html"  # Example file path

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

        # Connect to the database
        conn = connect_to_db()
        if conn:
            try:
                # Extract unique labels from Settings and Results
                labels = set(label for label, _ in data.get('Settings', []) + data.get('Results', []))

                # Create or alter run_performance table based on labels
                create_or_alter_table_run_performance(conn, labels)

                # Insert into run_performance first
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
