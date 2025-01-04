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
INITIAL_CASH = 1000  # Initial investment in euros

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
        self.order = None
        self.buy_price = None
        self.buy_date = None

    def next(self):
        if len(self) < max(self.p.resistance_period, self.p.ma_period, self.p.atr_period, self.p.macd_slow) + 1:
            return
        
        if self.order:
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
                            if not self.position:
                                self.buy()
                                self.buy_price = current_close
                                self.buy_date = current_date
                                self.breakouts.append((current_date, current_close))
        
        if self.position:
            if current_close >= self.buy_price * (1 + self.p.bull_run_threshold):
                self.close()
                self.bull_runs.append((self.buy_date, self.buy_price, current_date, current_close))
                gain = (current_close - self.buy_price) / self.buy_price * 100
                log_msg = f"Success: Breakout on {self.buy_date} at ${self.buy_price:.2f} led to bull run on {current_date} at ${current_close:.2f} (+{gain:.2f}%)"
                print(log_msg)
                logging.info(log_msg)
            elif len(self) - self.bar_executed > self.p.bull_run_period:
                self.close()
                loss = (current_close - self.buy_price) / self.buy_price * 100
                self.failures.append({
                    'Date': self.buy_date,
                    'Price': self.buy_price,
                    'Outcome': 'Failure',
                    'Reason': 'No bull run within period'
                })
                log_msg = f"Failure: Breakout on {self.buy_date} at ${self.buy_price:.2f} did not lead to a bull run within {self.p.bull_run_period} days. Current Price: ${current_close:.2f} ({loss:.2f}%)"
                print(log_msg)
                logging.info(log_msg)

def run_backtest():
    cerebro = bt.Cerebro()
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)
    cerebro.addstrategy(ImprovedBreakoutStrategy)
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission
    
    initial_value = cerebro.broker.getvalue()
    print(f'Starting Portfolio Value: {initial_value:.2f}')
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    print(f'Final Portfolio Value: {final_value:.2f}')
    
    return results[0], final_value

strategy, final_portfolio_value = run_backtest()

# Calculate buy and hold strategy returns
buy_and_hold_shares = INITIAL_CASH / data.iloc[0]['Close']
buy_and_hold_value = buy_and_hold_shares * data.iloc[-1]['Close']
buy_and_hold_return = (buy_and_hold_value - INITIAL_CASH) / INITIAL_CASH * 100

# Calculate strategy returns
strategy_return = (final_portfolio_value - INITIAL_CASH) / INITIAL_CASH * 100

print(f"Buy and Hold Strategy: {buy_and_hold_return:.2f}%")
print(f"Improved Breakout Strategy: {strategy_return:.2f}%")
print(f"Outperformance: {strategy_return - buy_and_hold_return:.2f}%")

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