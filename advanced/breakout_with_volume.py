import backtrader as bt
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Parameters
TICKER = 'AAPL'
START_DATE = '2019-01-01'
END_DATE = '2022-12-01'

# Fetch data using yfinance
data = yf.download(TICKER, start=START_DATE, end=END_DATE)
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Define the strategy
class ResistanceBreakoutStrategy(bt.Strategy):
    params = (
        ('resistance_period', 20),  # Lookback period for resistance
        ('volume_threshold', 1.5),  # Multiplier for average volume to consider as high
        ('bull_run_threshold', 0.05),  # 5% price increase to qualify as bull run
        ('bull_run_period', 5),  # Days within which the bull run should occur
    )
    
    def __init__(self):
        self.resistance = bt.ind.Highest(self.data.high, period=self.p.resistance_period)
        self.avg_volume = bt.ind.SMA(self.data.volume, period=self.p.resistance_period)
        self.breakouts = []
        self.bull_runs = []
    
    def next(self):
        # Check if current close breaks above resistance
        if self.data.close[0] > self.resistance[0]:
            # Check if volume is significantly higher than average
            if self.data.volume[0] > self.p.volume_threshold * self.avg_volume[0]:
                breakout_date = self.data.datetime.date(0)
                breakout_price = self.data.close[0]
                self.breakouts.append((breakout_date, breakout_price))
                
                # Check for bull run
                target_price = breakout_price * (1 + self.p.bull_run_threshold)
                for i in range(1, self.p.bull_run_period + 1):
                    if len(self.data) > i:
                        future_close = self.data.close[i]
                        if future_close >= target_price:
                            bull_run_date = self.data.datetime.date(i)
                            self.bull_runs.append((breakout_date, breakout_price, bull_run_date, future_close))
                            break

# Initialize Cerebro engine
cerebro = bt.Cerebro()

# Convert pandas DataFrame to Backtrader data feed
data_bt = bt.feeds.PandasData(dataname=data)

# Add data to Cerebro
cerebro.adddata(data_bt)

# Add the strategy
cerebro.addstrategy(ResistanceBreakoutStrategy)

# Set initial cash
cerebro.broker.setcash(100000.0)

# Run the backtest
strategies = cerebro.run()
strategy = strategies[0]

# Access the strategy's breakout and bull run data
breakouts = strategy.breakouts
bull_runs = strategy.bull_runs

# Print statistics
print(f"Total Breakouts: {len(breakouts)}")
print(f"Breakouts Leading to Bull Runs: {len(bull_runs)}")
if len(breakouts) > 0:
    success_rate = (len(bull_runs) / len(breakouts)) * 100
    print(f"Success Rate: {success_rate:.2f}%")
else:
    print("No breakouts detected.")

# Prepare data for plotting
plot_data = data.copy()
# To align the resistance array with the data, we need to pad the initial periods
resistance_values = strategy.resistance.array
padding = [None]*(len(plot_data) - len(resistance_values))
plot_data['Resistance'] = padding + list(resistance_values)

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

# Add bull run markers
for bull in bull_runs:
    fig.add_trace(go.Scatter(
        x=[bull[0]],
        y=[bull[1]],
        mode='markers',
        marker=dict(color='blue', size=10, symbol='star'),
        name='Bull Run Start'
    ))
    fig.add_trace(go.Scatter(
        x=[bull[2]],
        y=[bull[3]],
        mode='markers',
        marker=dict(color='red', size=10, symbol='star'),
        name='Bull Run Achieved'
    ))

# Update layout
fig.update_layout(
    title=f'Resistance Breakout Strategy for {TICKER}',
    yaxis_title='Price',
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
