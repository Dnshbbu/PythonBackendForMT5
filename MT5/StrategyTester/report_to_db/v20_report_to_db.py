
# Python code
import os
import re
import csv
import uuid
import psycopg2
from psycopg2 import sql
from bs4 import BeautifulSoup
import datetime
from parse_html import *
from create_CSVs import *
from save_to_DB import *
from create_plot import *


def extract_run_id_from_filename(html_file_path):
    file_name = os.path.basename(html_file_path)
    match = re.match(r"^(.*?)_report\.html$", file_name)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Filename '{file_name}' does not match the expected format '<run_id>_report.html'")


if __name__ == "__main__":
    # Replace with your actual HTML file path or modify to accept as an argument
    html_file_path = r"C:\Users\StdUser\Desktop\MyProjects\Backtesting\logs\INTC_10012324_report.html"  # Example file path
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

        # # Save Settings and Results to CSV
        # save_settings_and_results_to_csv(data, run_id)

        # Save Orders and Deals to CSV
        save_orders_and_deals_to_csv(data, run_id)

        create_graph(run_id)

        # Optionally, print all extracted data for verification
        # print("\nExtracted Data:")

        # # Print Settings
        # print("\n--- Settings ---")
        # for label, value in data.get('Settings', []):
        #     print(f"{label}: {value}")

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
