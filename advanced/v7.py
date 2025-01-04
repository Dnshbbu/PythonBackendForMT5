import backtrader as bt
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(filename='all_logs.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Parameters
TICKER = 'AAPL'
START_DATE = '2022-01-01'
END_DATE = '2022-06-01'

# Fetch data using yfinance
data = yf.download(TICKER, start=START_DATE, end=END_DATE)
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Display the first few rows to verify data
print(data.head())

# Define the strategy
class RefinedBreakoutStrategy(bt.Strategy):
    params = (
        ('resistance_period', 10),
        ('volume_multiplier', 1.1),
        ('bull_run_threshold', 0.03),
        ('bull_run_period', 10),
        ('rsi_period', 14),
        ('rsi_threshold', 55),
        ('ma_period', 50),
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('atr_period', 14),
        ('atr_threshold', 1.5),
    )
    
    def __init__(self):
        self.resistance = bt.ind.Highest(self.data.high(-1), period=self.p.resistance_period)
        self.avg_volume = bt.ind.SMA(self.data.volume, period=self.p.resistance_period)
        self.rsi = bt.ind.RSI_Safe(self.data.close, period=self.p.rsi_period)
        self.ma = bt.ind.SMA(self.data.close, period=self.p.ma_period)
        self.macd = bt.ind.MACD(self.data.close,
                                 period_me1=self.p.macd_fast,
                                 period_me2=self.p.macd_slow,
                                 period_signal=self.p.macd_signal)
        self.macd_cross = bt.ind.CrossOver(self.macd.macd, self.macd.signal)
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)
        
        self.breakouts = []
        self.bull_runs = []
        self.failures = []

    def next(self):
        required_length = max(self.p.resistance_period, self.p.ma_period, self.p.atr_period, self.p.macd_slow) + 1
        if len(self) < required_length:
            return
        
        current_close = self.data.close[0]
        current_volume = self.data.volume[0]
        current_resistance = self.resistance[0]
        current_avg_volume = self.avg_volume[0]
        current_rsi = self.rsi[0]
        current_ma = self.ma[0]
        current_macd = self.macd.macd[0]
        current_macd_signal = self.macd.signal[0]
        current_atr = self.atr[0]
        current_date = self.data.datetime.date(0)
        
        dynamic_volume_threshold = self.p.volume_multiplier * current_atr
        
        if current_close > current_resistance:
            if current_volume > dynamic_volume_threshold:
                if current_rsi > self.p.rsi_threshold:
                    if current_close > current_ma:
                        if current_macd > current_macd_signal:
                            breakout_date = current_date
                            breakout_price = current_close
                            self.breakouts.append((breakout_date, breakout_price))
                            
                            target_price = breakout_price * (1 + self.p.bull_run_threshold)
                            bull_run_achieved = False
                            bull_run_date = None
                            bull_run_price = None
                            
                            for i in range(1, self.p.bull_run_period + 1):
                                if len(self.data) > i:
                                    future_close = self.data.close[i]
                                    if future_close >= target_price:
                                        bull_run_date = self.data.datetime.date(i)
                                        bull_run_price = future_close
                                        self.bull_runs.append((breakout_date, breakout_price, bull_run_date, bull_run_price))
                                        bull_run_achieved = True
                                        break
                            
                            if bull_run_achieved:
                                gain = (bull_run_price - breakout_price) / breakout_price * 100
                                self.failures.append({
                                    'Breakout Date': breakout_date,
                                    'Breakout Price': breakout_price,
                                    'Outcome': 'Success',
                                    'Bull Run Date': bull_run_date,
                                    'Bull Run Price': bull_run_price,
                                    'Gain (%)': round(gain, 2),
                                    'Reason': 'Breakout confirmed by high volume, bullish RSI, MACD crossover, and MA uptrend.'
                                })
                                log_msg = f"Success: Breakout on {breakout_date} at ${breakout_price:.2f} led to bull run on {bull_run_date} at ${bull_run_price:.2f} (+{gain:.2f}%)"
                                print(log_msg)
                                logging.info(log_msg)
                            else:
                                last_price = self.data.close[-1]
                                loss = (last_price - breakout_price) / breakout_price * 100
                                self.failures.append({
                                    'Breakout Date': breakout_date,
                                    'Breakout Price': breakout_price,
                                    'Outcome': 'Failure',
                                    'Bull Run Date': None,
                                    'Bull Run Price': None,
                                    'Gain (%)': round(loss, 2),
                                    'Reason': 'Price did not reach target within the bull run period.'
                                })
                                log_msg = f"Failure: Breakout on {breakout_date} at ${breakout_price:.2f} did not lead to a bull run. Current Price: ${last_price:.2f} ({loss:.2f}%)"
                                print(log_msg)
                                logging.info(log_msg)
                        else:
                            self.record_failure(current_date, current_close, "MACD below signal line indicating bearish momentum.")
                    else:
                        self.record_failure(current_date, current_close, "Price below Moving Average indicating a downtrend.")
                else:
                    self.record_failure(current_date, current_close, "RSI below threshold indicating bearish momentum.")
            else:
                self.record_failure(current_date, current_close, "Volume not above dynamic threshold.")
        else:
            return
    
    def record_failure(self, date, price, reason):
        self.breakouts.append((date, price))
        last_price = self.data.close[-1]
        loss = (last_price - price) / price * 100
        self.failures.append({
            'Breakout Date': date,
            'Breakout Price': price,
            'Outcome': 'Failure',
            'Bull Run Date': None,
            'Bull Run Price': None,
            'Gain (%)': round(loss, 2),
            'Reason': reason
        })
        log_msg = f"Failure: Breakout on {date} at ${price:.2f} did not lead to a bull run. Current Price: ${last_price:.2f} ({loss:.2f}%)"
        print(log_msg)
        logging.info(log_msg)

# Initialize Cerebro engine
cerebro = bt.Cerebro()

# Create a data feed
data_feed = bt.feeds.PandasData(dataname=data)
cerebro.adddata(data_feed)

# Run backtesting and optimization until success rate >= 60%
success_rate = 0
iteration = 0
results_list = []

while success_rate < 60:
    iteration += 1
    cerebro.addstrategy(RefinedBreakoutStrategy)
    results = cerebro.run()
    strategy = results[0]

    # Calculate outcomes
    total_breakouts = len(strategy.breakouts)
    total_successful_bull_runs = len(strategy.bull_runs)
    total_failures = len(strategy.failures)

    # Calculate success rate
    success_rate = (total_successful_bull_runs / total_breakouts) * 100 if total_breakouts > 0 else 0

    # Log results for this iteration
    results_list.append({
        'Iteration': iteration,
        'Total Breakouts': total_breakouts,
        'Total Successful Bull Runs': total_successful_bull_runs,
        'Total Failures': total_failures,
        'Success Rate': success_rate
    })

    with open('continual_refining.txt', 'a') as f:
        f.write(f"{iteration},{total_breakouts},{total_successful_bull_runs},{total_failures},{success_rate:.2f}\n")
    
    # Prepare failure log for printing
    failure_log = pd.DataFrame(strategy.failures)
    print(failure_log)

# Logging final outcomes
with open('comparision.txt', 'w') as f:
    f.write(f"Total Breakouts: {total_breakouts}\n")
    f.write(f"Total Successful Bull Runs: {total_successful_bull_runs}\n")
    f.write(f"Total Failures: {total_failures}\n")
    f.write(f"Success Rate: {success_rate:.2f}%\n")

# Save success and failure reasons
with open('success_and_failure_reason.txt', 'w') as f:
    for entry in strategy.failures:
        f.write(f"{entry['Outcome']},{entry['Breakout Date']},{entry['Breakout Price']},{entry['Gain (%)']},{entry['Reason']}\n")

# Plotting results using Plotly
dates = [b[0] for b in strategy.breakouts]
breakout_prices = [b[1] for b in strategy.breakouts]

# Prepare Plotly figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=dates, y=breakout_prices, mode='markers', name='Breakouts', marker=dict(color='red', size=10)))
fig.update_layout(title='Breakout Strategy Backtesting', xaxis_title='Date', yaxis_title='Price')
fig.show()

# Show success and failure log
print("Breakout Strategy Summary")
print(f"Total Breakouts: {total_breakouts}")
print(f"Total Successful Bull Runs: {total_successful_bull_runs}")
print(f"Total Failures: {total_failures}")
print(f"Success Rate: {success_rate:.2f}%")
