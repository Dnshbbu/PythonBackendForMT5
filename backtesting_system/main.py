# main.py

import logging
from logging.handlers import RotatingFileHandler
from backtest import fetch_stock_data, run_backtest
from strategies import BuyAndHold, SMA_CrossOver_with_Indicators, BreakoutBullRun, SimpleCrossOver
from database import create_performance_table, save_run_results_to_db
import pandas as pd  # Add this to fix the "pd is not defined" issue

def setup_logging():
    """Sets up logging to both console and a rotating file."""
    logger = logging.getLogger()  # Get the root logger
    logger.setLevel(logging.INFO)  # Set the logger to capture only INFO and above

    # Create a formatter for consistent log output
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Check if handlers are already added to avoid duplication
    if not logger.hasHandlers():
        # Console handler (outputs to console)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Display INFO and above on console
        console_handler.setFormatter(formatter)

        # Rotating file handler (outputs to a file)
        file_handler = RotatingFileHandler('backtest.log', maxBytes=5*1024*1024, backupCount=5)
        file_handler.setLevel(logging.INFO)  # Log INFO and above to the file
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

setup_logging()

def run_strategy(stock_ticker, start_date, end_date, strategy_class, optim=False):
    logging.info(f"Running strategy: {strategy_class.__name__} for {stock_ticker} from {start_date} to {end_date}")
    
    # Create the performance table if it doesn't exist
    create_performance_table()

    # Fetch stock data as DataFrame and Backtrader Feed
    stock_df, stock_bt_feed = fetch_stock_data(stock_ticker, start_date, end_date)

    # Run backtest with Backtrader Feed
    backtest_results = run_backtest(stock_bt_feed, strategy_class, stock_ticker, optim=optim)

    if optim:
        # backtest_results is a DataFrame with optimization results
        # Save all optimization results to the database
        for index, row in backtest_results.iterrows():
            # Prepare run data excluding non-parameter fields if necessary
            run_data = {
                'fast_period': row.get('fast_period'),
                'slow_period': row.get('slow_period'),
                'stop_loss': row.get('stop_loss'),
                'final_balance': row.get('final_balance'),
                'returns': row.get('returns'),
                'lookback_period': row.get('lookback_period'),               # For BreakoutBullRun
                'exit_lookback_period': row.get('exit_lookback_period'),     # For BreakoutBullRun
                'hold_period': row.get('hold_period'),                       # For BreakoutBullRun
            }
            
            # Remove keys with NaN values (parameters not used by certain strategies)
            run_data = {k: v for k, v in run_data.items() if pd.notnull(v)}
            
            save_run_results_to_db(
                run_data=run_data,
                stock=stock_ticker,
                start_date=start_date,
                end_date=end_date,
                chart_html='',  # No chart for each optimization run
                strategy_class_name=strategy_class.__name__
            )
    else:
        # Extract strategy parameters
        # strategy_params = {k: v for k, v in strategy_class.params._getitems().items() if k != 'ticker'}
        strategy_params = {k: v for k, v in dict(strategy_class.params._getitems()).items() if k != 'ticker'}



        # The chart_html is now generated within the strategy, so we don't need to create it here
        chart_html = backtest_results.get('chart_path', '')
    
        # Save results to the database
        save_run_results_to_db(
            run_data=backtest_results,
            stock=stock_ticker,
            start_date=start_date,
            end_date=end_date,
            chart_html=chart_html,
            strategy_class_name=strategy_class.__name__
        )

    logging.info(f"Backtest {'optimization' if optim else 'completed'} for strategy: {strategy_class.__name__}")
    return backtest_results

def run_backtest_pipeline(stock='SNAP', start_date='2009-01-01', end_date='2022-04-20', strategy_name='SimpleCrossOver', optimize=False):
    """Function to run the backtest pipeline from another script by passing parameters."""
    
    # Map strategy names to strategy classes
    strategies_dict = {
        # 'BuyAndHold': BuyAndHold,
        # 'SMACrossOver': SMACrossOver,
        # 'SMA_CrossOver_with_Indicators': SMA_CrossOver_with_Indicators,
        # 'BreakoutBullRun': BreakoutBullRun,
        'SimpleCrossOver': SimpleCrossOver
    }
    
    strategy = strategies_dict.get(strategy_name)
    if strategy is None:
        logging.error(f"Strategy {strategy_name} not recognized.")
        raise ValueError(f"Strategy {strategy_name} not recognized.")
    
    if optimize:
        logging.info(f"\nRunning optimization for {strategy.__name__} Strategy:")
    else:
        logging.info(f"\nRunning {strategy.__name__} Strategy:")
    
    try:
        results = run_strategy(stock, start_date, end_date, strategy, optim=optimize)
        logging.info(f"Results: {results}")
        return results
    except Exception as e:
        logging.error(f"An error occurred while running {strategy.__name__}: {e}", exc_info=True)
        raise e

# Example call within the script itself, can be commented out
# results = run_backtest_pipeline(stock='SNAP', start_date='2019-01-01', end_date='2022-04-20', strategy_name='SimpleCrossOver', optimize=True)
if __name__ == '__main__':
    # Example call within the script itself, can be commented out
    results = run_backtest_pipeline(stock='SNAP', start_date='2019-01-01', end_date='2022-04-20', strategy_name='SimpleCrossOver', optimize=True)