# backtest.py

import backtrader as bt
import pandas as pd  # Import pandas
import yfinance as yf  # Import yfinance
import logging
from config import BACKTEST_CONFIG

def fetch_stock_data(stock_ticker, start_date, end_date):
    """Fetches stock data from Yahoo Finance as a DataFrame and Backtrader Feed."""
    logging.info(f"Fetching data for {stock_ticker} from {start_date} to {end_date}")
    
    # Fetch data as DataFrame using yfinance
    df = yf.download(stock_ticker, start=start_date, end=end_date)
    if df.empty:
        logging.error(f"No data fetched for {stock_ticker}. Please check the ticker and date range.")
        raise ValueError(f"No data fetched for {stock_ticker}.")
    logging.info(f"Fetched {len(df)} rows of data for {stock_ticker}.")

    # Convert DataFrame to Backtrader Feed
    df_bt = bt.feeds.PandasData(dataname=df)
    return df, df_bt

def run_backtest(stock_data_feed, strategy_class):
    """Sets up and runs the backtest using Backtrader."""
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class)

    # Add data to Backtrader
    cerebro.adddata(stock_data_feed)

    # Set starting cash
    cerebro.broker.setcash(BACKTEST_CONFIG['start_cash'])

    # Set commission
    cerebro.broker.setcommission(commission=BACKTEST_CONFIG['commission'])

    logging.info("Starting Backtest")
    # Run the backtest
    initial_balance = cerebro.broker.getvalue()
    logging.info(f"Starting Portfolio Value: {initial_balance:.2f}")
    cerebro.run()
    final_balance = cerebro.broker.getvalue()
    logging.info(f"Final Portfolio Value: {final_balance:.2f}")

    # Calculate returns and profit/loss
    # returns = (final_balance - initial_balance) / initial_balance
    
    profit_loss = final_balance - initial_balance
    returns = (profit_loss / initial_balance) * 100  # Convert to percentage

    return {
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'returns': returns,
        'profit_loss': profit_loss
    }
