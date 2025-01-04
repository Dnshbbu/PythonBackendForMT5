# main.py

import logging
from logging.handlers import RotatingFileHandler
from backtest import fetch_stock_data, run_backtest
from strategies import BuyAndHold, SMACrossOver, SMA_CrossOver_with_Indicators, BreakoutBullRun
from database import create_performance_table, save_run_results_to_db
from utils.plotting import create_interactive_plot

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

def run_strategy(stock_ticker, start_date, end_date, strategy_class):
    logging.info(f"Running strategy: {strategy_class.__name__} for {stock_ticker} from {start_date} to {end_date}")
    
    # Create the performance table if it doesn't exist
    create_performance_table()

    # Fetch stock data as DataFrame and Backtrader Feed
    stock_df, stock_bt_feed = fetch_stock_data(stock_ticker, start_date, end_date)

    # Run backtest with Backtrader Feed
    backtest_results = run_backtest(stock_bt_feed, strategy_class)

    # Extract strategy parameters and convert to dict
    strategy_params = dict(strategy_class.params._getitems())
    signals = results[0].get_signals() if hasattr(results[0], 'get_signals') else None

    # Create chart using the DataFrame and pass the strategy name
    chart_html = create_interactive_plot(
        stock_data=stock_df,
        ticker=stock_ticker,
        strategy_params=strategy_params,
        strategy_name=strategy_class.__name__,  # Pass strategy name
        signals=signals
    )

    # Save results to the database
    save_run_results_to_db(
        run_data=backtest_results,
        stock=stock_ticker,
        start_date=start_date,
        end_date=end_date,
        chart_html=chart_html,
        strategy_class_name=strategy_class.__name__
    )

    logging.info(f"Backtest completed for strategy: {strategy_class.__name__}")
    return backtest_results

if __name__ == "__main__":
    # Define the strategies you want to run
    strategies_to_run = [
        # BuyAndHold,
        # SMACrossOver,
        # SMA_CrossOver_with_Indicators
        BreakoutBullRun
    ]

    stock = 'SNAP'
    start_date = '2019-01-01'
    end_date = '2022-04-20'

    for strategy in strategies_to_run:
        logging.info(f"\nRunning {strategy.__name__} Strategy:")
        try:
            results = run_strategy(stock, start_date, end_date, strategy)
            logging.info(f"Results: {results}")
        except Exception as e:
            logging.error(f"An error occurred while running {strategy.__name__}: {e}", exc_info=True)
