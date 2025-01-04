import os
import re
import psycopg2
from psycopg2 import sql
from bs4 import BeautifulSoup

# Database connection parameters
DB_CONFIG = {
    'host': 'ep-morning-tree-a6vcm1gg.us-west-2.retooldb.com',
    'port': 5432,
    'dbname': 'retool',
    'user': 'retool',
    'password': 'gC5bAfzS7yXc',
}

def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        raise

def print_unique_settings_labels(settings):
    unique_labels = set(label for label, _ in settings)
    print(f"Unique Settings Labels ({len(unique_labels)}):")
    for label in sorted(unique_labels):
        print(f"- {label}")

# def create_run_performance_table(conn, settings):
#     unique_labels = set(label for label, _ in settings)

#     create_table_query = """
#     CREATE TABLE IF NOT EXISTS run_performance (
#         run_id VARCHAR PRIMARY KEY,
#     """

#     for label in unique_labels:
#         column_name = label.lower().replace(' ', '_').replace('/', '_').replace('-', '_')
#         create_table_query += f"    {column_name} TEXT,\n"

#     create_table_query += "    inputs TEXT\n);"

#     try:
#         with conn.cursor() as cur:
#             cur.execute(create_table_query)
#         conn.commit()
#         print("run_performance table created or verified successfully.")
#     except psycopg2.Error as e:
#         conn.rollback()
#         print(f"Error creating run_performance table: {e}")
#         raise


def create_run_performance_table(conn, settings):
    unique_labels = set(label for label, _ in settings)

    create_table_query = """
    CREATE TABLE IF NOT EXISTS run_performance (
        run_id VARCHAR PRIMARY KEY,
    """

    for label in unique_labels:
        # Replace special characters and convert to snake_case
        column_name = re.sub(r'[^a-zA-Z0-9]+', '_', label.lower()).strip('_')
        create_table_query += f"    {column_name} TEXT,\n"

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
    create_run_performance_table(conn, settings)

    create_other_tables = [
        """
        CREATE TABLE IF NOT EXISTS orders (
            order_id SERIAL PRIMARY KEY,
            run_id VARCHAR REFERENCES run_performance(run_id),
            open_time VARCHAR,
            "order" VARCHAR,
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
    file_name = os.path.basename(html_file_path)
    match = re.match(r"^(.*?)_report\.html$", file_name, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Filename '{file_name}' does not match the expected format '<run_id>_report.html'")

def parse_strategy_tester_report(html_file):
    try:
        with open(html_file, 'r', encoding='utf-16', errors='replace') as file:
            soup = BeautifulSoup(file, 'html.parser')
    except Exception as e:
        print(f"Failed to read the file with UTF-16 encoding: {e}")
        return {}

    tables = soup.find_all('table')

    settings_data = []
    orders_data = []
    deals_data = []

    for table_index, table in enumerate(tables, start=1):
        print(f"\n{'='*40}\nParsing Table {table_index}\n{'='*40}")
        rows = table.find_all('tr')

        if table_index == 1:
            current_section = 'Settings'
        else:
            current_section = None

        is_collecting_inputs = False
        inputs_list = []

        for row in rows:
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

            if row.find('img'):
                continue

            cells = row.find_all('td')

            if not cells:
                continue

            if table_index == 1 and current_section == 'Settings':
                if is_collecting_inputs:
                    label_cell_text = cells[0].get_text(strip=True)
                    if label_cell_text.endswith(':') and label_cell_text.lower() != 'inputs:':
                        if inputs_list:
                            concatenated_inputs = '; '.join(inputs_list)
                            settings_data.append(('Inputs', concatenated_inputs))
                            inputs_list = []
                        is_collecting_inputs = False

                    if is_collecting_inputs:
                        value_cell = cells[-1]
                        value = value_cell.get_text(strip=True)
                        if value:
                            inputs_list.append(value)
                        continue

                extracted_pairs = extract_key_value_pairs(cells)

                for label, value in extracted_pairs:
                    label_lower = label.lower()
                    if label_lower == 'inputs':
                        is_collecting_inputs = True
                        inputs_list.append(value)
                    else:
                        settings_data.append((label, value))
            elif table_index > 1 and current_section in ['Orders', 'Deals']:
                row_data = [cell.get_text(strip=True) for cell in cells]
                print(f"{current_section} Row: {row_data}")
                
                if all(cell == '' for cell in row_data):
                    continue

                if current_section == 'Orders':
                    if len(row_data) >= 11:
                        orders_data.append(row_data[:11])
                    else:
                        print(f"Skipped incomplete Orders row: {row_data}")
                elif current_section == 'Deals':
                    if len(row_data) >= 13:
                        deals_data.append(row_data[:13])
                    else:
                        print(f"Skipped incomplete Deals row: {row_data}")

        if table_index == 1 and current_section == 'Settings' and is_collecting_inputs and inputs_list:
            concatenated_inputs = '; '.join(inputs_list)
            settings_data.append(('Inputs', concatenated_inputs))

        print(f"Finished parsing Table {table_index}\n")

    return {
        'Settings': settings_data,
        'Orders': orders_data,
        'Deals': deals_data
    }

def extract_key_value_pairs(cells):
    pairs = []
    i = 0
    while i < len(cells):
        cell_text = cells[i].get_text(strip=True)
        if cell_text.endswith(':'):
            label = cell_text.rstrip(':').strip()
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
        i += 1
    return pairs

def insert_run_performance(conn, run_id, settings):
    data = {label: value for label, value in settings}

    inputs = data.pop('Inputs', None)

    columns = ['run_id']
    values = [run_id]

    for key, value in data.items():
        column = key.lower().replace(' ', '_').replace('/', '_').replace('-', '_')
        columns.append(column)
        values.append(value)

    if inputs:
        columns.append('inputs')
        values.append(inputs)

    print(f"Columns to insert: {columns}")
    print(f"Values to insert: {values}")

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
    if not orders:
        print("No Orders data to insert.")
        return

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
    if not deals:
        print("No Deals data to insert.")
        return

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
    try:
        insert_run_performance(conn, run_id, data.get('Settings', []))
        
        insert_orders(conn, run_id, data.get('Orders', []))
        
        insert_deals(conn, run_id, data.get('Deals', []))
        
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

            print_unique_settings_labels(data.get('Settings', []))

            conn = get_db_connection()
            print("Database connection established.")

            create_tables(conn, data.get('Settings', []))

            process_and_insert_data(conn, run_id, data)

        except Exception as e:
            print(f"An error occurred during processing: {e}")
            raise
        finally:
            if 'conn' in locals() and conn:
                conn.close()
                print("Database connection closed.")