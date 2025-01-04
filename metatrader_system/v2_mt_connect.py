import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

# Initialize MetaTrader 5
if not mt5.initialize():
    print("Failed to initialize MT5, error code:", mt5.last_error())
    quit()

# Set symbol and time frame
symbol = "TSLA"
timeframe = mt5.TIMEFRAME_D1  # D1 stands for daily time frame
days = 100

# Ensure Tesla (TSLA) is available in the market watch
if not mt5.symbol_select(symbol, True):
    print(f"Failed to select {symbol}")
    mt5.shutdown()
    quit()

# Set the date range (100 days back from today)
today = datetime.now()
start_date = today - timedelta(days=days)

# Request the data
rates = mt5.copy_rates_range(symbol, timeframe, start_date, today)

# Shutdown connection to MetaTrader 5
# mt5.shutdown()

# # Convert the data to a pandas DataFrame
# if rates is not None:
#     data = pd.DataFrame(rates)
#     # Convert the 'time' column to datetime format for better readability
#     data['time'] = pd.to_datetime(data['time'], unit='s')

#     # Print the last 100 days of Tesla stock data
#     print(data)
# else:
#     print(f"No data found for {symbol} in the specified range")

# account_info = mt5.account_info()
account_info = mt5.account_info()
if account_info is None:
    print("Failed to get account information, error code:", mt5.last_error())
    mt5.shutdown()
    quit()
else:
    print("Account Info:")
    print(account_info)
