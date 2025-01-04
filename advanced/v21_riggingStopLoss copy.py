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
        self.rsi = bt.ind.RSI(self.data.close, period=self.p.rsi_period)  # Use standard RSI
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
    Returns the initial cash, final portfolio value, and the strategy's performance metrics.
    """
    cerebro = bt.Cerebro()
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)

    cerebro.addstrategy(ImprovedBreakoutStrategy, 
                         resistance_period=params[0], 
                         volume_multiplier=params[1], 
                         trailing_stop_atr_multiplier=params[2])

    cerebro.broker.setcash(INITIAL_CASH)

    logging.info(f"Starting Portfolio Value: {cerebro.broker.getvalue()}")

    cerebro.run()
    
    final_value = cerebro.broker.getvalue()
    logging.info(f"Ending Portfolio Value: {final_value}")
    
    return INITIAL_CASH, final_value

# --- 6. Execute the Backtest ---
if __name__ == "__main__":
    # Define the ranges for parameters
    resistance_periods = range(5, 21)  # 5 to 20 days
    volume_multipliers = np.arange(1.0, 2.1, 0.1)  # 1.0 to 2.0
    trailing_stop_multipliers = np.arange(1.0, 3.1, 0.1)  # 1.0 to 3.0

    # Generate parameter combinations
    param_combinations = list(product(resistance_periods, volume_multipliers, trailing_stop_multipliers))

    # Use multiprocessing to run backtests in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(partial(run_backtest, data=data), param_combinations)

    # --- 7. Log final results ---
    logging.info("Backtest completed. Results:")
    for initial_cash, final_value in results:
        logging.info(f"Initial Cash: {initial_cash}, Final Value: {final_value}")
