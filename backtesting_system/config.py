# config.py

from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env file

DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'port': os.getenv('DB_PORT')
}

BACKTEST_CONFIG = {
    'start_cash': 10000,
    'commission': 0.001  # 0.1% commission
}

# Future enhancements can be added here
