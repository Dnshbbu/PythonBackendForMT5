
# Python code
import os
import re
import csv
import uuid
import psycopg2
from psycopg2 import sql
from bs4 import BeautifulSoup
import datetime



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

