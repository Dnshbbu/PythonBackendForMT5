
# Python code
import os
import re
import csv
import uuid
import psycopg2
from psycopg2 import sql
from bs4 import BeautifulSoup
import datetime



# Database connection parameters
DB_CONFIG = {
    'host': 'ep-morning-tree-a6vcm1gg.us-west-2.retooldb.com',  # e.g., 'localhost'
    'port': 5432,                # Default PostgreSQL port
    'dbname': 'retool',          # Your database name
    'user': 'retool',            # Your database username
    'password': 'gC5bAfzS7yXc',  # Your database password
}


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
        'comments': '',  # Adding the comments field
        'created_at': datetime.datetime.now()  # Adding the current datetime
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
    columns = ['uuid', 'run_id', 'comments', 'created_at'] + fixed_columns_order
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

    # time_columns = [
    #     'minimal_position_holding_time', 'maximal_position_holding_time',
    #     'average_position_holding_time'
    # ]

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
    # elif column in time_columns:
    #     # Convert to HH:MM:SS format
    #     if isinstance(value, str):
    #         try:
    #             parts = value.strip().split(':')
    #             if len(parts) == 3:
    #                 return f"{int(parts[0]):02}:{int(parts[1]):02}:{int(parts[2]):02}"
    #             else:
    #                 return None
    #         except:
    #             return None
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
        ('minimal_position_holding_time', 'VARCHAR(255)'),
        ('maximal_position_holding_time', 'VARCHAR(255)'),
        ('average_position_holding_time', 'VARCHAR(255)'),
        ('created_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),  # Add created_at column
        ('comments', 'TEXT')
    ]

    create_or_alter_table(conn, 'run_performance', required_columns)
