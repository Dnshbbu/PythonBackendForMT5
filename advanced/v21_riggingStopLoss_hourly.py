import pandas as pd
import numpy as np
import talib as ta
import matplotlib.pyplot as plt

# Assuming your dataset is loaded as df with 'datetime', 'open', 'high', 'low', 'close', and 'volume' columns
# Ensure 'datetime' is in proper datetime format
df['datetime'] = pd.to_datetime(df['datetime'])

# Resample to hourly data (if necessary)
df = df.set_index('datetime').resample('H').ffill().dropna()  # Resample to hourly frequency, forward fill missing data

# Parameters (may need tuning for hourly data)
resistance_period = 13
volume_multiplier = 1.0
trailing_stop_atr_multiplier = 1.5
stop_loss_atr_multiplier = 2.0

# Define functions to calculate indicators

def calculate_atr(df, period=14):
    """Calculate the Average True Range (ATR)"""
    high = df['high']
    low = df['low']
    close = df['close']
    atr = ta.ATR(high, low, close, timeperiod=period)
    return atr

def calculate_resistance(df, period=13):
    """Identify resistance levels based on the highest high of a given period"""
    return df['high'].rolling(window=period).max()

# Add indicators to the dataframe
df['ATR'] = calculate_atr(df, period=14)
df['Resistance'] = calculate_resistance(df, period=resistance_period)

# Trading logic
initial_balance = 10000
balance = initial_balance
position = None
trades = []

for i in range(resistance_period, len(df)):
    # Breakout condition
    current_price = df['close'].iloc[i]
    resistance = df['Resistance'].iloc[i]
    atr = df['ATR'].iloc[i]
    
    if current_price > resistance and position is None:
        # Buy signal
        position = {
            'entry_price': current_price,
            'stop_loss': current_price - stop_loss_atr_multiplier * atr,
            'trailing_stop': current_price - trailing_stop_atr_multiplier * atr,
            'entry_time': df.index[i]
        }
        print(f"Bought on {df.index[i]} at ${current_price:.2f} with initial stop-loss set at ${position['stop_loss']:.2f} and trailing stop set at ${position['trailing_stop']:.2f}")
    
    elif position:
        # Update trailing stop
        new_trailing_stop = max(position['trailing_stop'], current_price - trailing_stop_atr_multiplier * atr)
        position['trailing_stop'] = new_trailing_stop
        print(f"Updated trailing stop to ${new_trailing_stop:.2f} on {df.index[i]}")
        
        # Exit condition: trailing stop hit
        if current_price < position['trailing_stop']:
            profit_loss = (current_price - position['entry_price']) / position['entry_price'] * 100
            balance += profit_loss / 100 * initial_balance  # Update balance based on profit/loss
            trades.append({
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'entry_time': position['entry_time'],
                'exit_time': df.index[i],
                'profit_loss': profit_loss
            })
            print(f"Trailing Stop: Sold on {df.index[i]} at ${current_price:.2f} (Trailing Stop: ${position['trailing_stop']:.2f}, Gain: {profit_loss:.2f}%)")
            position = None  # Reset position

# After loop: print final results and analysis
print(f"Final balance: ${balance:.2f}")
print(f"Total trades: {len(trades)}")

# Simple performance metrics
total_return = (balance - initial_balance) / initial_balance * 100
print(f"Strategy Return: {total_return:.2f}%")

# Buy-and-hold strategy (for comparison)
buy_and_hold_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
print(f"Buy and Hold Return: {buy_and_hold_return:.2f}%")
print(f"Outperformance: {total_return - buy_and_hold_return:.2f}%")

# Plot trades (buy/sell points)
df['close'].plot(figsize=(12, 6), label='Price')
for trade in trades:
    plt.scatter(trade['entry_time'], trade['entry_price'], color='green', marker='^', label='Buy')
    plt.scatter(trade['exit_time'], trade['exit_price'], color='red', marker='v', label='Sell')
plt.legend()
plt.title('Breakout Strategy with Hourly Data')
plt.show()

# Save trade summary and visualization
trade_summary = pd.DataFrame(trades)
trade_summary.to_csv('trade_summary.csv')
