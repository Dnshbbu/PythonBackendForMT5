import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# 1. Initialize MetaTrader 5
if not mt5.initialize():
    print("Failed to initialize MT5, error code:", mt5.last_error())
    quit()
else:
    print("MT5 initialized successfully.")

# 2. Retrieve Account Information (Optional but recommended)
account_info = mt5.account_info()
if account_info is None:
    print("Failed to get account information, error code:", mt5.last_error())
    mt5.shutdown()
    quit()
else:
    print(f"Account Number: {account_info.login}")
    print(f"Leverage: {account_info.leverage}")
    print(f"Balance: {account_info.balance}\n")

# 3. Define Symbol and Timeframe
symbol = "TSLA"  # Adjust this if your broker uses a different symbol name
timeframe = mt5.TIMEFRAME_D1  # Daily timeframe
days = 100  # Number of days to backtest

# 4. Verify Symbol Availability
if not mt5.symbol_select(symbol, True):
    print(f"Symbol '{symbol}' not found. Attempting to find alternative symbols...")
    # List all available symbols containing 'TSLA'
    available_symbols = [s.name for s in mt5.symbols_get() if "TSLA" in s.name.upper()]
    if available_symbols:
        symbol = available_symbols[0]
        print(f"Using alternative symbol: {symbol}")
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to select symbol '{symbol}'.")
            mt5.shutdown()
            quit()
    else:
        print(f"No alternative symbols found for '{symbol}'.")
        mt5.shutdown()
        quit()
else:
    print(f"Symbol '{symbol}' selected successfully.\n")

# 5. Define Date Range (Use UTC Time)
today = datetime.utcnow()
start_date = today - timedelta(days=days)
print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')} (UTC)\n")

# 6. Request Historical Data
rates = mt5.copy_rates_range(symbol, timeframe, start_date, today)

# 7. Check Data Retrieval
if rates is None or len(rates) == 0:
    print(f"No data found for {symbol} in the specified range. Error code:", mt5.last_error())
    mt5.shutdown()
    quit()

# 8. Convert Data to pandas DataFrame
data = pd.DataFrame(rates)
data['time'] = pd.to_datetime(data['time'], unit='s')
data.set_index('time', inplace=True)
data.sort_index(inplace=True)  # Ensure data is sorted by date

print("Historical Data Retrieved Successfully:\n")
print(data.tail())  # Display the last few rows for verification

# 9. Perform Buy and Hold Backtest
initial_price = data['open'].iloc[0]
final_price = data['close'].iloc[-1]
initial_investment = 1000  # Example: $1,000
units = initial_investment / initial_price
final_value = units * final_price
total_return = (final_value - initial_investment) / initial_investment * 100

print("\n--- Buy and Hold Backtest Results ---")
print(f"Buy at: {data.index[0].date()} at price: ${initial_price:.2f}")
print(f"Sell at: {data.index[-1].date()} at price: ${final_price:.2f}")
print(f"Initial Investment: ${initial_investment:.2f}")
print(f"Final Value: ${final_value:.2f}")
print(f"Total Return: {total_return:.2f}%")

# 10. (Optional) Plot the Price Chart with Buy and Hold
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['close'], label='Close Price', color='blue')
plt.scatter(data.index[0], initial_price, color='green', label='Buy', marker='^', s=100)
plt.scatter(data.index[-1], final_price, color='red', label='Sell', marker='v', s=100)
plt.title(f'Buy and Hold Strategy for {symbol}')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()

# 11. Shutdown MT5 Connection
mt5.shutdown()
print("\nMT5 connection shut down successfully.")
