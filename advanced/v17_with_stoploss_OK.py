import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import logging
from itertools import product
import csv

# Configure logging
logging.basicConfig(filename='backtest_logs.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Parameters
TICKER = 'AAPL'
START_DATE = '2019-01-01'
END_DATE = '2020-06-01'
INITIAL_CASH = 1000  # Initial investment in euros

# Fetch data using yfinance
data = yf.download(TICKER, start=START_DATE, end=END_DATE)
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

class ImprovedBreakoutStrategy(bt.Strategy):
    params = (
        ('resistance_period', range(5, 16, 2)),  # Optimized
        ('volume_multiplier', np.arange(1.0, 1.5, 0.1)),  # Optimized
        ('bull_run_threshold', np.arange(0.02, 0.05, 0.01)),  # Optimized
        ('bull_run_period', range(5, 16, 2)),  # Optimized (bars)
        ('rsi_period', 14),
        ('rsi_threshold', 55),
        ('ma_period', 50),
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('atr_period', 14),
        ('atr_threshold', 1.5),
        ('stop_loss_atr_multiplier', 2.0),  # New parameter for stop-loss
    )

    def __init__(self):
        self.resistance = bt.ind.Highest(self.data.high(-1), period=self.p.resistance_period)
        self.avg_volume = bt.ind.SMA(self.data.volume, period=self.p.resistance_period)
        self.rsi = bt.ind.RSI_Safe(self.data.close, period=self.p.rsi_period)
        self.ma = bt.ind.SMA(self.data.close, period=self.p.ma_period)
        self.macd = bt.ind.MACD(self.data.close, period_me1=self.p.macd_fast,
                                 period_me2=self.p.macd_slow, period_signal=self.p.macd_signal)
        self.macd_cross = bt.ind.CrossOver(self.macd.macd, self.macd.signal)
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)

        self.breakouts = []
        self.bull_runs = []
        self.failures = []
        self.stop_losses = []  # To track stop-loss events
        self.order = None
        self.buy_price = None
        self.buy_date = None
        self.buy_bar_count = 0
        self.stop_price = None  # To store the stop-loss price

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

        if not self.position:
            # Check for breakout conditions to enter a trade
            if current_close > current_resistance:
                if current_volume > dynamic_volume_threshold:
                    if current_rsi > self.p.rsi_threshold:
                        if current_close > current_ma:
                            if current_macd > current_macd_signal:
                                self.buy()
                                self.buy_price = current_close
                                self.buy_date = current_date
                                self.buy_bar_count = 0
                                self.breakouts.append((current_date, current_close))
                                # Set stop-loss price based on ATR
                                self.stop_price = self.buy_price - (self.p.stop_loss_atr_multiplier * current_atr)
                                log_msg = f"Bought on {self.buy_date} at ${self.buy_price:.2f} with stop-loss set at ${self.stop_price:.2f}"
                                print(log_msg)
                                logging.info(log_msg)
        else:
            # Manage the open position
            self.buy_bar_count += 1

            # Check for bull run (take profit)
            if current_close >= self.buy_price * (1 + self.p.bull_run_threshold):
                self.close()
                self.bull_runs.append((self.buy_date, self.buy_price, current_date, current_close))
                gain = (current_close - self.buy_price) / self.buy_price * 100
                log_msg = f"Success: Breakout on {self.buy_date} at ${self.buy_price:.2f} led to bull run on {current_date} at ${current_close:.2f} (+{gain:.2f}%)"
                print(log_msg)
                logging.info(log_msg)
                self.stop_price = None  # Reset stop-loss after exiting
            # Check for stop-loss
            elif current_close <= self.stop_price:
                self.close()
                loss = (current_close - self.buy_price) / self.buy_price * 100
                self.stop_losses.append({
                    'Date': self.buy_date,
                    'Price': self.buy_price,
                    'Stop Price': self.stop_price,
                    'Exit Date': current_date,
                    'Exit Price': current_close,
                    'Loss (%)': loss
                })
                log_msg = f"Stop-Loss: Sold on {current_date} at ${current_close:.2f} (Stop Price: ${self.stop_price:.2f}, Loss: {loss:.2f}%)"
                print(log_msg)
                logging.info(log_msg)
                self.stop_price = None  # Reset stop-loss after exiting
            # Check for time-based exit (no bull run within period)
            elif self.buy_bar_count > self.p.bull_run_period:
                self.close()
                loss = (current_close - self.buy_price) / self.buy_price * 100
                self.failures.append({
                    'Date': self.buy_date,
                    'Price': self.buy_price,
                    'Outcome': 'Failure',
                    'Reason': 'No bull run within period',
                    'Exit Date': current_date,
                    'Exit Price': current_close,
                    'Loss (%)': loss
                })
                log_msg = f"Failure: Breakout on {self.buy_date} at ${self.buy_price:.2f} did not lead to a bull run within {self.p.bull_run_period} days. Sold on {current_date} at ${current_close:.2f} ({loss:.2f}%)"
                print(log_msg)
                logging.info(log_msg)
                self.stop_price = None  # Reset stop-loss after exiting

def run_backtest(params=None):
    cerebro = bt.Cerebro()
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)

    cerebro.addstrategy(ImprovedBreakoutStrategy, **params)  # Add strategy with parameters here

    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001)

    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    return final_value, results[0]  # Return both final value and the strategy instance

# Optimization Loop
best_params = {}
best_return = float('-inf')
optimization_results = []

# Generate parameter combinations
param_combinations = list(product(
    ImprovedBreakoutStrategy.params.resistance_period,
    ImprovedBreakoutStrategy.params.volume_multiplier,
    ImprovedBreakoutStrategy.params.bull_run_threshold,
    ImprovedBreakoutStrategy.params.bull_run_period,
    [2.0]  # Fixed stop_loss_atr_multiplier or include in optimization
))
total_iterations = len(param_combinations)

# Open the CSV file for writing iteration results
with open('iteration_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['Iteration', 'Parameters', 'Return']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i, (resistance_period, volume_multiplier, bull_run_threshold, bull_run_period, stop_loss_atr_multiplier) in enumerate(param_combinations):
        params = {
            'resistance_period': resistance_period,
            'volume_multiplier': volume_multiplier,
            'bull_run_threshold': bull_run_threshold,
            'bull_run_period': bull_run_period,
            'stop_loss_atr_multiplier': stop_loss_atr_multiplier,  # Include stop-loss parameter
        }

        final_value, strategy = run_backtest(params)
        strategy_return = (final_value - INITIAL_CASH) / INITIAL_CASH
        optimization_results.append((params, strategy_return, strategy))

        # Write iteration results to CSV
        writer.writerow({
            'Iteration': f"{i + 1}/{total_iterations}",
            'Parameters': params,
            'Return': f"{strategy_return:.4f}"
        })

        print(f"Iteration {i + 1}/{total_iterations}: Parameters: {params}, Return: {strategy_return:.4f}")

        if strategy_return > best_return:
            best_return = strategy_return
            best_params = params
            best_strategy = strategy
            print(f"New best parameters found: {best_params}, Return: {best_return:.4f}")

print(f"\nOptimization complete. Best parameters: {best_params}, Best return: {best_return:.4f}")

# --- Save Final Results and Analysis to a Separate File ---

with open('final_results.txt', 'w') as f:
    # Redirect print output to the file
    import sys
    original_stdout = sys.stdout
    sys.stdout = f 

    print(f"Optimization complete. Best parameters: {best_params}, Best return: {best_return:.4f}")

    # Calculate buy and hold strategy returns (no change required here)
    buy_and_hold_shares = INITIAL_CASH / data.iloc[0]['Close']
    buy_and_hold_value = buy_and_hold_shares * data.iloc[-1]['Close']
    buy_and_hold_return = (buy_and_hold_value - INITIAL_CASH) / INITIAL_CASH * 100

    # Calculate strategy returns using best_strategy 
    final_value = best_return * INITIAL_CASH + INITIAL_CASH  # Approximation
    strategy_return = (final_value - INITIAL_CASH) / INITIAL_CASH * 100

    print(f"Buy and Hold Strategy: {buy_and_hold_return:.2f}%")
    print(f"Improved Breakout Strategy: {strategy_return:.2f}%")
    print(f"Outperformance: {strategy_return - buy_and_hold_return:.2f}%")

    logging.info(f"Total Breakouts: {len(best_strategy.breakouts)}")
    logging.info(f"Total Successful Bull Runs: {len(best_strategy.bull_runs)}")
    logging.info(f"Total Failures: {len(best_strategy.failures)}")
    logging.info(f"Total Stop-Loss Exits: {len(best_strategy.stop_losses)}")

    failure_log = pd.DataFrame(best_strategy.failures)
    print(failure_log)

    stop_loss_log = pd.DataFrame(best_strategy.stop_losses)
    print(stop_loss_log)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))

    for breakout in best_strategy.breakouts:  # Access breakouts from best_strategy
        fig.add_trace(go.Scatter(x=[breakout[0]], y=[breakout[1]], mode='markers',
                                 marker=dict(color='red', size=10), name='Breakout'))

    for bull_run in best_strategy.bull_runs:
        fig.add_trace(go.Scatter(x=[bull_run[2]], y=[bull_run[3]], mode='markers',
                                 marker=dict(color='green', size=10), name='Bull Run'))

    for stop_loss in best_strategy.stop_losses:
        fig.add_trace(go.Scatter(x=[stop_loss['Exit Date']], y=[stop_loss['Exit Price']],
                                 mode='markers', marker=dict(color='orange', size=10), name='Stop-Loss'))

    fig.update_layout(title='Breakouts, Bull Runs, and Stop-Loss Exits (Optimized)', 
                      xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=True)
    # fig.show() # Do not show the plot, as it will cause issues when redirecting output

    print(f"Strategy identified {len(best_strategy.breakouts)} breakouts, "
          f"{len(best_strategy.bull_runs)} bull runs, and {len(best_strategy.stop_losses)} stop-loss exits.")
    success_rate = len(best_strategy.bull_runs) / len(best_strategy.breakouts) * 100 if best_strategy.breakouts else 0
    print(f"Success rate: {success_rate:.2f}%")

    # --- Optimization Results Visualization ---
    returns = [r for _, r, _ in optimization_results]  # Unpack returns, ignoring strategy instances
    fig = go.Figure(data=[go.Histogram(x=returns, nbinsx=20)])
    fig.update_layout(title='Distribution of Returns from Optimization',
                      xaxis_title='Returns', yaxis_title='Frequency')
    # fig.show() # Do not show the plot

    sys.stdout = original_stdout  # Reset print output back to console
