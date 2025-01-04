import os
import re
import psycopg2
from psycopg2 import sql
from bs4 import BeautifulSoup

# Database connection parameters
DB_CONFIG = {
    'host': 'ep-morning-tree-a6vcm1gg.us-west-2.retooldb.com',         # e.g., 'localhost'
    'port': 5432,                # Default PostgreSQL port
    'dbname': 'retool',   # Your database name
    'user': 'retool',     # Your database username
    'password': 'gC5bAfzS7yXc', # Your database password
}

def get_db_connection():
    """
    Establishes and returns a connection to the PostgreSQL database.
    
    Returns:
        conn: psycopg2 connection object
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False  # Enable transaction management
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        raise

def print_unique_settings_labels(settings):
    unique_labels = set(label for label, _ in settings)
    print(f"Unique Settings Labels ({len(unique_labels)}):")
    for label in sorted(unique_labels):
        print(f"- {label}")

def create_run_performance_table(conn, settings):
    """
    Creates the run_performance table with dynamic columns based on settings.
    
    Args:
        conn: psycopg2 connection object
        settings (list of tuples): List of (label, value) for settings
    """
    # Extract unique labels
    unique_labels = set(label for label, _ in settings)

    # Start building the CREATE TABLE statement
    create_table_query = """
    CREATE TABLE IF NOT EXISTS run_performance (
        run_id VARCHAR PRIMARY KEY,
    """

    # Define data types based on sample values (simplistic approach)
    # You might need to adjust data types based on actual data
    for label in unique_labels:
        column_name = label.lower().replace(' ', '_').replace('/', '_').replace('-', '_')
        # Simplistic type assignment: all text. Adjust as needed.
        create_table_query += f"    {column_name} TEXT,\n"

    # Add 'inputs' column
    create_table_query += "    inputs TEXT\n);"

    try:
        with conn.cursor() as cur:
            cur.execute(create_table_query)
        conn.commit()
        print("run_performance table created or verified successfully.")
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error creating run_performance table: {e}")
        raise

def create_tables(conn, settings):
    """
    Creates the required tables in the PostgreSQL database if they do not exist.
    
    Args:
        conn: psycopg2 connection object
        settings (list of tuples): List of (label, value) for settings
    """
    create_run_performance_table(conn, settings)

    create_other_tables = [
        """
        CREATE TABLE IF NOT EXISTS orders (
            order_id SERIAL PRIMARY KEY,
            run_id VARCHAR REFERENCES run_performance(run_id),
            open_time VARCHAR,
            order VARCHAR,
            symbol VARCHAR,
            type VARCHAR,
            volume VARCHAR,
            price VARCHAR,
            s_l VARCHAR,
            t_p VARCHAR,
            time VARCHAR,
            state VARCHAR,
            comment VARCHAR
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS deals (
            deal_id SERIAL PRIMARY KEY,
            run_id VARCHAR REFERENCES run_performance(run_id),
            time VARCHAR,
            deal VARCHAR,
            symbol VARCHAR,
            type VARCHAR,
            direction VARCHAR,
            volume VARCHAR,
            price VARCHAR,
            order_field VARCHAR,
            commission VARCHAR,
            swap VARCHAR,
            profit VARCHAR,
            balance VARCHAR,
            comment VARCHAR
        );
        """
    ]
    
    try:
        with conn.cursor() as cur:
            for query in create_other_tables:
                cur.execute(query)
        conn.commit()
        print("Orders and Deals tables created or verified successfully.")
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error creating Orders or Deals tables: {e}")
        raise

def extract_run_id_from_filename(html_file_path):
    """
    Extracts the run_id from the HTML file name.
    The expected format is <run_id>_report.html.
    
    Args:
        html_file_path (str): Path to the HTML file

    Returns:
        str: Extracted run_id
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
        html_file (str): Path to the HTML file

    Returns:
        dict: Extracted data categorized into 'Settings', 'Results', 'Orders', and 'Deals'
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

def insert_run_performance(conn, run_id, settings, results):
    """
    Inserts data into the run_performance table.

    Args:
        conn: psycopg2 connection object
        run_id (str): The run ID
        settings (list of tuples): List of (label, value) for settings
        results (list of tuples): List of (label, value) for results
    """
    # Combine settings and results into a dictionary
    data = {label: value for label, value in settings}
    data.update({label: value for label, value in results})

    # Remove 'Inputs' as it's already included in settings
    inputs = data.pop('Inputs', None)

    # Prepare columns and values
    columns = ['run_id']
    values = [run_id]

    for key, value in data.items():
        # Convert keys to snake_case to match database column naming conventions
        column = key.lower().replace(' ', '_').replace('/', '_').replace('-', '_')
        columns.append(column)
        values.append(value)

    if inputs:
        columns.append('inputs')
        values.append(inputs)

    # Debugging: Print columns and values
    print(f"Columns to insert: {columns}")
    print(f"Values to insert: {values}")

    # Construct the INSERT statement dynamically
    insert_query = sql.SQL("""
        INSERT INTO run_performance ({fields})
        VALUES ({values})
        ON CONFLICT (run_id) DO UPDATE SET {update_fields};
    """).format(
        fields=sql.SQL(', ').join(map(sql.Identifier, columns)),
        values=sql.SQL(', ').join(sql.Placeholder() * len(values)),
        update_fields=sql.SQL(', ').join([
            sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(col), sql.Identifier(col))
            for col in columns if col != 'run_id'
        ])
    )

    try:
        with conn.cursor() as cur:
            cur.execute(insert_query, values)
        print(f"Inserted/Updated run_performance for run_id: {run_id}")
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error inserting into run_performance: {e.pgcode} - {e.pgerror}")
        raise

def insert_orders(conn, run_id, orders):
    """
    Inserts data into the orders table.

    Args:
        conn: psycopg2 connection object
        run_id (str): The run ID
        orders (list of lists): List of orders data
    """
    if not orders:
        print("No Orders data to insert.")
        return

    # Define the insert query with placeholders
    insert_query = """
        INSERT INTO orders (
            run_id,
            open_time,
            order,
            symbol,
            type,
            volume,
            price,
            s_l,
            t_p,
            time,
            state,
            comment
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """

    # Prepare the data with run_id
    orders_data = [
        [run_id] + order for order in orders
    ]

    try:
        with conn.cursor() as cur:
            cur.executemany(insert_query, orders_data)
        print(f"Inserted {len(orders)} rows into orders table.")
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error inserting into orders: {e.pgcode} - {e.pgerror}")
        raise

def insert_deals(conn, run_id, deals):
    """
    Inserts data into the deals table.

    Args:
        conn: psycopg2 connection object
        run_id (str): The run ID
        deals (list of lists): List of deals data
    """
    if not deals:
        print("No Deals data to insert.")
        return

    # Define the insert query with placeholders
    insert_query = """
        INSERT INTO deals (
            run_id,
            time,
            deal,
            symbol,
            type,
            direction,
            volume,
            price,
            order_field,
            commission,
            swap,
            profit,
            balance,
            comment
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """

    # Prepare the data with run_id
    deals_data = [
        [run_id] + deal for deal in deals
    ]

    try:
        with conn.cursor() as cur:
            cur.executemany(insert_query, deals_data)
        print(f"Inserted {len(deals)} rows into deals table.")
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error inserting into deals: {e.pgcode} - {e.pgerror}")
        raise

def process_and_insert_data(conn, run_id, data):
    """
    Processes the parsed data and inserts it into the database.

    Args:
        conn: psycopg2 connection object
        run_id (str): The run ID
        data (dict): Parsed data containing 'Settings', 'Results', 'Orders', and 'Deals'
    """
    try:
        # Insert run_performance data
        insert_run_performance(conn, run_id, data.get('Settings', []), data.get('Results', []))
        
        # Insert orders data
        insert_orders(conn, run_id, data.get('Orders', []))
        
        # Insert deals data
        insert_deals(conn, run_id, data.get('Deals', []))
        
        # Commit all changes
        conn.commit()
        print("All data inserted successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Error during data insertion: {e}")
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

            # Print unique settings labels
            print_unique_settings_labels(data.get('Settings', []))

            # Establish database connection
            conn = get_db_connection()
            print("Database connection established.")

            # Create tables if they don't exist, using settings
            create_tables(conn, data.get('Settings', []))

            # Insert data into the database
            process_and_insert_data(conn, run_id, data)

        except Exception as e:
            print(f"An error occurred during processing: {e}")
            raise
        finally:
            if 'conn' in locals() and conn:
                conn.close()
                print("Database connection closed.")
