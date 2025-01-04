import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define parameters for the strategy
LOOKBACK_PERIOD = 10  # Look back to find resistance/support
EXIT_LOOKBACK_PERIOD = 5  # Look back for exiting position
STOP_LOSS_PERCENTAGE = 0.95  # Stop loss as a percentage of the entry price
HOLD_PERIOD = 10  # Hold resistance/support for this number of bars
TICKER = "SNAP"  # Change to your desired ticker

# Initialize MetaTrader 5
if not mt5.initialize():
    logging.error("initialize() failed")
    mt5.shutdown()

# Function to get historical data
def get_historical_data(symbol, timeframe, num_bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None:
        logging.error(f"Failed to get historical data for {symbol}")
        return None
    return pd.DataFrame(rates)

# Main trading logic
def run_strategy():
    resistance = None
    support = None
    bar_count = 0
    stop_loss_price = 0
    position_ticket = None
    
    while True:
        # Get current price and historical data
        data = get_historical_data(TICKER, mt5.TIMEFRAME_M1, HOLD_PERIOD + LOOKBACK_PERIOD + EXIT_LOOKBACK_PERIOD)
        if data is None:
            break

        close_prices = data['close'].values
        highest_high = np.max(close_prices[-LOOKBACK_PERIOD:])
        lowest_low = np.min(close_prices[-EXIT_LOOKBACK_PERIOD:])

        # Update resistance and support
        if bar_count >= HOLD_PERIOD or resistance is None:
            resistance = highest_high
            support = lowest_low
            bar_count = 0
            logging.info(f"Updated Resistance: {resistance}, Support: {support}")

        bar_count += 1
        
        # Get current market price
        current_price = mt5.symbol_info_tick(TICKER).ask

        # Log current prices and static resistance/support levels
        logging.info(f"Current Price: {current_price:.5f}, Static Resistance: {resistance}, Static Support: {support}")

        # Check for open positions
        positions = mt5.positions_get(symbol=TICKER)
        if positions:
            position_ticket = positions[0].ticket

            # Check for sell signal
            if current_price < support or current_price < stop_loss_price:
                result = mt5.order_send(symbol=TICKER, action=mt5.ORDER_SELL, volume=0.1, price=current_price, sl=0, tp=0, comment="Sell Order")
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.info("SELL ORDER EXECUTED at {:.5f}".format(current_price))
                position_ticket = None  # Reset position ticket after selling
        else:
            # Check for buy signal
            if current_price > resistance:
                stop_loss_price = current_price * STOP_LOSS_PERCENTAGE
                result = mt5.order_send(symbol=TICKER, action=mt5.ORDER_BUY, volume=0.1, price=current_price, sl=stop_loss_price, tp=0, comment="Buy Order")
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.info("BUY ORDER EXECUTED at {:.5f}".format(current_price))
        
        # Sleep to avoid high-frequency trading (optional)
        time.sleep(60)

# Run the strategy
try:
    run_strategy()
except KeyboardInterrupt:
    logging.info("Trading stopped.")
finally:
    mt5.shutdown()
