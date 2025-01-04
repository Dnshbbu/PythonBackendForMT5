import backtrader as bt
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Parameters
TICKER = 'AAPL'
START_DATE = '2019-01-01'
END_DATE = '2022-06-01'

# Fetch data using yfinance
data = yf.download(TICKER, start=START_DATE, end=END_DATE)
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Display the first few rows to verify data
print(data.head())

# Define the strategy
class ResistanceBreakoutStrategy(bt.Strategy):
    params = (
        ('resistance_period', 10),        # Lookback period for resistance
        ('volume_threshold', 1.2),        # Multiplier for average volume to consider as high
        ('bull_run_threshold', 0.03),     # 3% price increase to qualify as bull run
        ('bull_run_period', 7),           # Days within which the bull run should occur
    )
    
    def __init__(self):
        # Calculate the highest high over the previous 'resistance_period' bars, excluding current bar
        self.resistance = bt.ind.Highest(self.data.high(-1), period=self.p.resistance_period)
        
        # Calculate the Simple Moving Average (SMA) of volume over the 'resistance_period'
        self.avg_volume = bt.ind.SMA(self.data.volume, period=self.p.resistance_period)
        
        # Initialize lists to store breakout and bull run information
        self.breakouts = []
        self.bull_runs = []
    
    def next(self):
        # Ensure there is enough data to compute indicators
        if len(self) < self.p.resistance_period + 1:
            return
        
        current_close = self.data.close[0]
        current_volume = self.data.volume[0]
        current_resistance = self.resistance[0]
        current_avg_volume = self.avg_volume[0]
        current_date = self.data.datetime.date(0)
        
        # Debugging: Print current status (optional)
        # Uncomment the next line to see detailed logs
        # print(f"Date: {current_date}, Close: {current_close}, Resistance: {current_resistance}, Volume: {current_volume}, Avg Volume: {current_avg_volume}")
        
        # Check if current close breaks above the resistance level
        if current_close > current_resistance:
            # Check if volume is significantly higher than average
            if current_volume > self.p.volume_threshold * current_avg_volume:
                breakout_date = current_date
                breakout_price = current_close
                self.breakouts.append((breakout_date, breakout_price))
                
                # Define the target price for the bull run
                target_price = breakout_price * (1 + self.p.bull_run_threshold)
                
                # Initialize bull run variables
                bull_run_achieved = False
                bull_run_date = None
                bull_run_price = None
                
                # Iterate through the next 'bull_run_period' bars to check for bull run
                for i in range(1, self.p.bull_run_period + 1):
                    if len(self.data) > i:
                        future_close = self.data.close[i]
                        if future_close >= target_price:
                            bull_run_date = self.data.datetime.date(i)
                            bull_run_price = future_close
                            self.bull_runs.append((breakout_date, breakout_price, bull_run_date, bull_run_price))
                            bull_run_achieved = True
                            break
                
                # Logging breakout and bull run information
                if bull_run_achieved:
                    print(f"Breakout on {breakout_date} at ${breakout_price:.2f} led to bull run on {bull_run_date} at ${bull_run_price:.2f}")
                else:
                    print(f"Breakout on {breakout_date} at ${breakout_price:.2f} did not lead to a bull run within {self.p.bull_run_period} days.")

# Initialize Cerebro engine
cerebro = bt.Cerebro()

# Convert pandas DataFrame to Backtrader data feed
data_bt = bt.feeds.PandasData(dataname=data)

# Add data to Cerebro
cerebro.adddata(data_bt)

# Add the strategy
cerebro.addstrategy(ResistanceBreakoutStrategy)

# Set initial cash (optional, as we're not trading)
cerebro.broker.setcash(100000.0)

# Run the backtest
strategies = cerebro.run()
strategy = strategies[0]

# Access the strategy's breakout and bull run data
breakouts = strategy.breakouts
bull_runs = strategy.bull_runs

# Print statistics
print("\n=== Backtest Statistics ===")
print(f"Total Breakouts Detected: {len(breakouts)}")
print(f"Breakouts Leading to Bull Runs: {len(bull_runs)}")
if len(breakouts) > 0:
    success_rate = (len(bull_runs) / len(breakouts)) * 100
    print(f"Success Rate: {success_rate:.2f}%")
else:
    print("No breakouts detected.")

# Prepare data for plotting
plot_data = data.copy()

# Align the resistance array with the data by shifting it forward by 1 to exclude current bar
resistance_values = strategy.resistance.array
# Calculate padding length
padding_length = len(plot_data) - len(resistance_values)
if padding_length > 0:
    plot_data['Resistance'] = [None]*padding_length + list(resistance_values)
else:
    plot_data['Resistance'] = list(resistance_values)

# Create Plotly figure
fig = go.Figure()

# Add candlestick
fig.add_trace(go.Candlestick(
    x=plot_data.index,
    open=plot_data['Open'],
    high=plot_data['High'],
    low=plot_data['Low'],
    close=plot_data['Close'],
    name='Price'
))

# Add resistance line
fig.add_trace(go.Scatter(
    x=plot_data.index,
    y=plot_data['Resistance'],
    mode='lines',
    line=dict(color='orange', dash='dash'),
    name='Resistance'
))

# Add breakout markers
for breakout in breakouts:
    fig.add_trace(go.Scatter(
        x=[breakout[0]],
        y=[breakout[1]],
        mode='markers',
        marker=dict(color='green', size=10, symbol='triangle-up'),
        name='Breakout'
    ))

# Add bull run start markers (blue stars)
for bull in bull_runs:
    fig.add_trace(go.Scatter(
        x=[bull[0]],
        y=[bull[1]],
        mode='markers',
        marker=dict(color='blue', size=10, symbol='star'),
        name='Bull Run Start'
    ))

# Add bull run achieved markers (red stars)
for bull in bull_runs:
    fig.add_trace(go.Scatter(
        x=[bull[2]],
        y=[bull[3]],
        mode='markers',
        marker=dict(color='red', size=10, symbol='star'),
        name='Bull Run Achieved'
    ))

# Update layout
fig.update_layout(
    title=f'Resistance Breakout Strategy for {TICKER} ({START_DATE} to {END_DATE})',
    yaxis_title='Price (USD)',
    xaxis_title='Date',
    xaxis_rangeslider_visible=False,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

# Show the plot
fig.show()
