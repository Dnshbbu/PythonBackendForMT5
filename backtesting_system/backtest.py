# backtest.py

import backtrader as bt
import pandas as pd
import yfinance as yf
import logging
from config import BACKTEST_CONFIG
import os

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

def run_backtest(stock_data_feed, strategy_class, stock_ticker, optim=False):
    """Sets up and runs the backtest using Backtrader."""
    cerebro = bt.Cerebro(optreturn=False)  # Set optreturn=True if you want to collect optimization results

    if optim:
        # Check if the strategy class has 'opt_params'
        if hasattr(strategy_class, 'opt_params'):
            opt_params = strategy_class.opt_params
            cerebro.optstrategy(strategy_class, **opt_params)
            logging.info(f"Optimization parameters for {strategy_class.__name__}: {opt_params}")
        else:
            logging.error(f"Strategy {strategy_class.__name__} does not have 'opt_params' defined for optimization.")
            raise AttributeError(f"Strategy {strategy_class.__name__} does not have 'opt_params' defined for optimization.")
    else:
        # Add strategy normally
        cerebro.addstrategy(strategy_class, ticker=stock_ticker)
    
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
    results = cerebro.run()
    final_balance = cerebro.broker.getvalue()
    logging.info(f"Final Portfolio Value: {final_balance:.2f}")
    
    if optim:
        # When optim=True, results is a list of lists (each inner list corresponds to a strategy instance)
        optimized_results = []
        for strat_list in results:
            for strat in strat_list:
                # strat is a strategy instance
                final_value = strat.broker.getvalue()
                # params = strat.params._getitems()
                params = dict(strat.params._getitems())  # Convert to dict
                optimized_results.append({
                    **params,
                    'final_balance': final_value,
                    'returns': (final_value - initial_balance) / initial_balance * 100
                })
        
        # Convert to DataFrame for easier analysis
        df_optim = pd.DataFrame(optimized_results)
        # Sort by returns or final_balance
        df_optim_sorted = df_optim.sort_values(by='returns', ascending=False)
        
        # Get the best parameters
        best_params = df_optim_sorted.iloc[0]
        logging.info("Optimization Results:")
        logging.info(df_optim_sorted)
        logging.info(f"Best Parameters: {best_params}")
        
        # Save optimization results to CSV
        output_dir = "optimization_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        csv_file = os.path.join(output_dir, f"{strategy_class.__name__}_optimization_results.csv")
        df_optim_sorted.to_csv(csv_file, index=False)
        logging.info(f"Optimization results saved to {csv_file}")
        
        return df_optim_sorted
    else:
        # Calculate returns and profit/loss
        profit_loss = final_balance - initial_balance
        returns = (profit_loss / initial_balance) * 100  # Convert to percentage
        
        # Get the chart path from the strategy (assuming it's stored in the strategy)
        chart_path = getattr(results[0], 'chart_path', None)
        
        return {
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'returns': returns,
            'profit_loss': profit_loss,
            'chart_path': chart_path
        }
