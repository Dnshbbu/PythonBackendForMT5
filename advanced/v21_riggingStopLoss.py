import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import logging
from itertools import product
import csv
import multiprocessing
from functools import partial

# --- 1. Configure Logging ---
logging.basicConfig(
    filename='backtest_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- 2. Define Parameters ---
TICKER = 'SNAP'
START_DATE = '2019-01-01'
END_DATE = '2022-06-01'
INITIAL_CASH = 1000  # Initial investment in euros

# --- 3. Fetch Data ---
def fetch_data(ticker, start, end):
    """
    Fetches historical stock data from Yahoo Finance.
    """
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        logging.error(f"No data fetched for {ticker} from {start} to {end}.")
        raise ValueError(f"No data fetched for {ticker} from {start} to {end}.")
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

data = fetch_data(TICKER, START_DATE, END_DATE)

# --- 4. Define the Trading Strategy ---
class ImprovedBreakoutStrategy(bt.Strategy):
    params = (
        ('resistance_period', 10),  # Default value; will be optimized
        ('volume_multiplier', 1.2),  # Default value; will be optimized
        ('trailing_stop_atr_multiplier', 1.0),  # Default value; will be optimized
        ('stop_loss_atr_multiplier', 2.0),  # Fixed stop-loss multiplier
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
        self.macd = bt.ind.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )
        self.macd_cross = bt.ind.CrossOver(self.macd.macd, self.macd.signal)
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)

        # Tracking variables
        self.breakouts = []
        self.stop_losses = []  # To track stop-loss events
        self.trailing_stops = []  # To track trailing stop exits
        self.immediate_exits = []  # To track immediate exits
        self.order = None
        self.buy_price = None
        self.buy_date = None
        self.trailing_stop = None  # To store the trailing stop price

    def next(self):
        # Ensure enough data is available
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
                                self.breakouts.append((current_date, current_close))
                                # Set initial stop-loss price based on ATR
                                self.stop_price = self.buy_price - (self.p.stop_loss_atr_multiplier * current_atr)
                                # Initialize trailing stop
                                self.trailing_stop = self.buy_price - (self.p.trailing_stop_atr_multiplier * current_atr)
                                log_msg = (
                                    f"Bought on {self.buy_date} at ${self.buy_price:.2f} with "
                                    f"initial stop-loss set at ${self.stop_price:.2f} and "
                                    f"trailing stop set at ${self.trailing_stop:.2f}"
                                )
                                print(log_msg)
                                logging.info(log_msg)
        else:
            # Manage the open position
            current_atr = self.atr[0]  # Update ATR in case it changes
            current_close = self.data.close[0]

            # --- Immediate Exit Condition ---
            if current_close < self.buy_price:
                self.close()
                loss = 0.0  # As per user's request, record exit price as buy price, implying zero loss
                self.immediate_exits.append({
                    'Date': self.buy_date,
                    'Price': self.buy_price,
                    'Exit Date': current_date,
                    'Exit Price': self.buy_price,  # Record as buy price
                    'Loss (%)': loss  # Zero loss
                })
                log_msg = (
                    f"Immediate Exit: Sold on {current_date} at ${self.buy_price:.2f} "
                    f"(Bought at: ${self.buy_price:.2f}, Loss: {loss:.2f}%)"
                )
                print(log_msg)
                logging.info(log_msg)
                self.trailing_stop = None  # Reset trailing stop after exiting
                return  # Exit early since position is closed

            # --- Update Trailing Stop ---
            if current_close > self.buy_price:
                # Calculate new trailing stop
                new_trailing_stop = current_close - (self.p.trailing_stop_atr_multiplier * current_atr)
                if new_trailing_stop > self.trailing_stop:
                    self.trailing_stop = new_trailing_stop
                    log_msg = f"Updated trailing stop to ${self.trailing_stop:.2f} on {current_date}"
                    print(log_msg)
                    logging.info(log_msg)

            # --- Check for Trailing Stop ---
            if current_close <= self.trailing_stop:
                self.close()
                gain = (current_close - self.buy_price) / self.buy_price * 100
                self.trailing_stops.append({
                    'Date': self.buy_date,
                    'Price': self.buy_price,
                    'Trailing Stop': self.trailing_stop,
                    'Exit Date': current_date,
                    'Exit Price': current_close,
                    'Gain (%)': gain
                })
                log_msg = (
                    f"Trailing Stop: Sold on {current_date} at ${current_close:.2f} "
                    f"(Trailing Stop: ${self.trailing_stop:.2f}, Gain: {gain:.2f}%)"
                )
                print(log_msg)
                logging.info(log_msg)
                self.trailing_stop = None  # Reset trailing stop after exiting

            # --- Check for Initial Stop-Loss ---
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
                log_msg = (
                    f"Stop-Loss: Sold on {current_date} at ${current_close:.2f} "
                    f"(Stop Price: ${self.stop_price:.2f}, Loss: {loss:.2f}%)"
                )
                print(log_msg)
                logging.info(log_msg)
                self.trailing_stop = None  # Reset trailing stop after exiting

# --- 5. Define the Backtest Function ---
def run_backtest(params, data):
    """
    Runs a backtest with the given parameters and data.
    Returns the final portfolio value and the strategy's performance metrics.
    """
    cerebro = bt.Cerebro()
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)

    cerebro.addstrategy(ImprovedBreakoutStrategy, **params)

    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001)

    try:
        results = cerebro.run()
        final_value = cerebro.broker.getvalue()
        strategy = results[0]
        # Calculate return
        strategy_return = (final_value - INITIAL_CASH) / INITIAL_CASH
        return (final_value, strategy_return, strategy)
    except Exception as e:
        logging.error(f"Error running backtest with params {params}: {e}")
        return (None, None, None)

# --- 6. Define the Worker Function for Parallel Processing ---
def worker(params, data):
    """
    Worker function to run backtest in parallel.
    """
    final_value, strategy_return, strategy = run_backtest(params, data)
    if final_value is not None and strategy_return is not None:
        return (params, strategy_return)
    else:
        return (params, None)

# --- 7. Main Optimization Function ---
def optimize_strategy(data):
    """
    Optimizes the trading strategy by testing various parameter combinations in parallel.
    """
    best_params = {}
    best_return = float('-inf')
    optimization_results = []

    # Generate parameter combinations
    resistance_periods = range(5, 16, 2)  # e.g., 5, 7, 9, ..., 15
    volume_multipliers = np.arange(1.0, 1.5, 0.1)  # e.g., 1.0, 1.1, ..., 1.4
    trailing_stop_atr_multipliers = [1.0, 1.5, 2.0]  # Example values; adjust as needed

    param_combinations = list(product(
        resistance_periods,
        volume_multipliers,
        trailing_stop_atr_multipliers
    ))
    total_iterations = len(param_combinations)

    # Prepare list of parameter dictionaries
    param_dicts = [{
        'resistance_period': resistance_period,
        'volume_multiplier': volume_multiplier,
        'trailing_stop_atr_multiplier': trailing_stop_atr_multiplier,
        'stop_loss_atr_multiplier': 2.0  # Fixed stop-loss multiplier; can also be optimized if desired
    } for (resistance_period, volume_multiplier, trailing_stop_atr_multiplier) in param_combinations]

    # --- Parallel Optimization ---
    cpu_count = multiprocessing.cpu_count()
    pool_size = cpu_count - 1 if cpu_count > 1 else 1  # Leave one CPU free
    print(f"Starting optimization using {pool_size} parallel processes...")

    with multiprocessing.Pool(processes=pool_size) as pool:
        # Partial function to fix the 'data' argument
        worker_func = partial(worker, data=data)
        # Map the worker function to the parameter dictionaries
        results = pool.map(worker_func, param_dicts)

    # --- Process Results ---
    # Open the CSV file for writing iteration results
    with open('iteration_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Iteration', 'Parameters', 'Return']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, result in enumerate(results):
            params, strategy_return = result
            if strategy_return is not None:
                optimization_results.append((params, strategy_return))

                # Write iteration results to CSV
                writer.writerow({
                    'Iteration': f"{i + 1}/{total_iterations}",
                    'Parameters': params,
                    'Return': f"{strategy_return:.4f}"
                })

                print(f"Iteration {i + 1}/{total_iterations}: Parameters: {params}, Return: {strategy_return:.4f}")

                # Update best parameters
                if strategy_return > best_return:
                    best_return = strategy_return
                    best_params = params
                    print(f"New best parameters found: {best_params}, Return: {best_return:.4f}")
            else:
                # Handle failed backtest
                writer.writerow({
                    'Iteration': f"{i + 1}/{total_iterations}",
                    'Parameters': params,
                    'Return': 'Error'
                })
                print(f"Iteration {i + 1}/{total_iterations}: Parameters: {params}, Return: Error")

    print(f"\nOptimization complete. Best parameters: {best_params}, Best return: {best_return:.4f}")

    # --- 8. Save Final Results and Analysis to a Separate File ---
    with open('final_results.txt', 'w') as f:
        # Redirect print output to the file
        import sys
        original_stdout = sys.stdout
        sys.stdout = f 

        print(f"Optimization complete. Best parameters: {best_params}, Best return: {best_return:.4f}")

        # Calculate buy and hold strategy returns
        buy_and_hold_shares = INITIAL_CASH / data.iloc[0]['Close']
        buy_and_hold_value = buy_and_hold_shares * data.iloc[-1]['Close']
        buy_and_hold_return = (buy_and_hold_value - INITIAL_CASH) / INITIAL_CASH * 100

        # Calculate strategy returns using best_params
        final_value, strategy_return, strategy = run_backtest(best_params, data)
        if final_value is not None:
            strategy_return_percentage = (final_value - INITIAL_CASH) / INITIAL_CASH * 100
        else:
            strategy_return_percentage = None

        print(f"Buy and Hold Strategy: {buy_and_hold_return:.2f}%")
        if strategy_return_percentage is not None:
            print(f"Improved Breakout Strategy: {strategy_return_percentage:.2f}%")
            print(f"Outperformance: {strategy_return_percentage - buy_and_hold_return:.2f}%")
        else:
            print("Improved Breakout Strategy: Failed to run backtest.")
            print("Outperformance: N/A")

        # Logging summary
        if strategy is not None:
            logging.info(f"Total Breakouts: {len(strategy.breakouts)}")
            logging.info(f"Total Stop-Loss Exits: {len(strategy.stop_losses)}")
            logging.info(f"Total Trailing Stop Exits: {len(strategy.trailing_stops)}")
            logging.info(f"Total Immediate Exits: {len(strategy.immediate_exits)}")

            # Print detailed logs
            failure_log = pd.DataFrame(strategy.failures) if hasattr(strategy, 'failures') else pd.DataFrame()
            print("\nFailures:")
            if not failure_log.empty:
                print(failure_log)
            else:
                print("No failures recorded.")

            stop_loss_log = pd.DataFrame(strategy.stop_losses)
            print("\nStop-Loss Exits:")
            if not stop_loss_log.empty:
                print(stop_loss_log)
            else:
                print("No stop-loss exits recorded.")

            trailing_stop_log = pd.DataFrame(strategy.trailing_stops)
            print("\nTrailing Stop Exits:")
            if not trailing_stop_log.empty:
                print(trailing_stop_log)
            else:
                print("No trailing stop exits recorded.")

            immediate_exit_log = pd.DataFrame(strategy.immediate_exits)
            print("\nImmediate Exits:")
            if not immediate_exit_log.empty:
                print(immediate_exit_log)
            else:
                print("No immediate exits recorded.")

            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))

            for breakout in strategy.breakouts:
                fig.add_trace(go.Scatter(
                    x=[breakout[0]], y=[breakout[1]],
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name='Breakout'
                ))

            for trailing_stop in strategy.trailing_stops:
                fig.add_trace(go.Scatter(
                    x=[trailing_stop['Exit Date']], y=[trailing_stop['Exit Price']],
                    mode='markers',
                    marker=dict(color='purple', size=10),
                    name='Trailing Stop'
                ))

            for stop_loss in strategy.stop_losses:
                fig.add_trace(go.Scatter(
                    x=[stop_loss['Exit Date']], y=[stop_loss['Exit Price']],
                    mode='markers',
                    marker=dict(color='orange', size=10),
                    name='Stop-Loss'
                ))

            for immediate_exit in strategy.immediate_exits:
                fig.add_trace(go.Scatter(
                    x=[immediate_exit['Exit Date']], y=[immediate_exit['Exit Price']],
                    mode='markers',
                    marker=dict(color='blue', size=10),
                    name='Immediate Exit'
                ))

            fig.update_layout(
                title='Breakouts, Trailing Stop Exits, Stop-Loss Exits, and Immediate Exits (Optimized)',
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=True
            )
            # Save the plot as an HTML file
            fig.write_html("breakouts_trailing_stops_immediate_exits.html")
            print("\nVisualization saved as 'breakouts_trailing_stops_immediate_exits.html'.")

            # Summary
            print(f"\nStrategy identified {len(strategy.breakouts)} breakouts, "
                  f"{len(strategy.stop_losses)} stop-loss exits, "
                  f"{len(strategy.trailing_stops)} trailing stop exits, and "
                  f"{len(strategy.immediate_exits)} immediate exits.")
            # Success rate based on trailing stops
            success_rate = (len(strategy.trailing_stops) / len(strategy.breakouts) * 100) if len(strategy.breakouts) > 0 else 0
            print(f"Success rate (Trailing Stops): {success_rate:.2f}%")

            # --- Optimization Results Visualization ---
            returns = [r for _, r in optimization_results if r is not None]
            fig_hist = go.Figure(data=[go.Histogram(x=returns, nbinsx=20)])
            fig_hist.update_layout(
                title='Distribution of Returns from Optimization',
                xaxis_title='Returns',
                yaxis_title='Frequency'
            )
            # Save the histogram as an HTML file
            fig_hist.write_html("returns_distribution.html")
            print("Returns distribution histogram saved as 'returns_distribution.html'.")

        else:
            print("Best strategy instance is not available.")

        sys.stdout = original_stdout  # Reset print output back to console

# --- 9. Entry Point ---
if __name__ == "__main__":
    optimize_strategy(data)
