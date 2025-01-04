import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(filename='backtest_logs.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Parameters
TICKER = 'SNAP'
START_DATE = '2019-01-01'
END_DATE = '2022-06-01'

# Fetch data using yfinance
data = yf.download(TICKER, start=START_DATE, end=END_DATE)
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

class ImprovedBreakoutStrategy(bt.Strategy):
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
        self.macd = bt.ind.MACD(self.data.close, period_me1=self.p.macd_fast, period_me2=self.p.macd_slow, period_signal=self.p.macd_signal)
        self.macd_cross = bt.ind.CrossOver(self.macd.macd, self.macd.signal)
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)
        
        self.breakouts = []
        self.bull_runs = []
        self.failures = []
    
    def next(self):
        if len(self) < max(self.p.resistance_period, self.p.ma_period, self.p.atr_period, self.p.macd_slow) + 1:
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
                                log_msg = f"Success: Breakout on {breakout_date} at ${breakout_price:.2f} led to bull run on {bull_run_date} at ${bull_run_price:.2f} (+{gain:.2f}%)"
                                print(log_msg)
                                logging.info(log_msg)
                            else:
                                last_price = self.data.close[-1]
                                loss = (last_price - breakout_price) / breakout_price * 100
                                self.failures.append({
                                    'Date': breakout_date,
                                    'Price': breakout_price,
                                    'Outcome': 'Failure',
                                    'Reason': 'No bull run within period'
                                })
                                log_msg = f"Failure: Breakout on {breakout_date} at ${breakout_price:.2f} did not lead to a bull run within {self.p.bull_run_period} days. Current Price: ${last_price:.2f} ({loss:.2f}%)"
                                print(log_msg)
                                logging.info(log_msg)
                        else:
                            self.failures.append({
                                'Date': current_date,
                                'Price': current_close,
                                'Outcome': 'Failure',
                                'Reason': 'MACD below signal'
                            })
                    else:
                        self.failures.append({
                            'Date': current_date,
                            'Price': current_close,
                            'Outcome': 'Failure',
                            'Reason': 'Price below MA'
                        })
                else:
                    self.failures.append({
                        'Date': current_date,
                        'Price': current_close,
                        'Outcome': 'Failure',
                        'Reason': 'RSI below threshold'
                    })
            else:
                self.failures.append({
                    'Date': current_date,
                    'Price': current_close,
                    'Outcome': 'Failure',
                    'Reason': 'Insufficient volume'
                })

def run_backtest():
    cerebro = bt.Cerebro()
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)
    cerebro.addstrategy(ImprovedBreakoutStrategy)
    results = cerebro.run()
    return results[0]

strategy = run_backtest()

logging.info(f"Total Breakouts: {len(strategy.breakouts)}")
logging.info(f"Total Successful Bull Runs: {len(strategy.bull_runs)}")
logging.info(f"Total Failures: {len(strategy.failures)}")

failure_log = pd.DataFrame(strategy.failures)
print(failure_log)

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))

for breakout in strategy.breakouts:
    fig.add_trace(go.Scatter(x=[breakout[0]], y=[breakout[1]], mode='markers', marker=dict(color='red', size=10), name='Breakout'))

for bull_run in strategy.bull_runs:
    fig.add_trace(go.Scatter(x=[bull_run[2]], y=[bull_run[3]], mode='markers', marker=dict(color='green', size=10), name='Bull Run'))

fig.update_layout(title='Breakouts and Bull Runs', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=True)
fig.show()

print(f"Strategy identified {len(strategy.breakouts)} breakouts and {len(strategy.bull_runs)} bull runs.")
success_rate = len(strategy.bull_runs) / len(strategy.breakouts) * 100 if strategy.breakouts else 0
print(f"Success rate: {success_rate:.2f}%")