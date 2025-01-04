# strategies/breakout_bull_run.py

import backtrader as bt
import logging


class BreakoutBullRun(bt.Strategy):
    params = (
        ('lookback_period', 10),  # Look back to find resistance/support
        ('exit_lookback_period', 5),  # Look back for exiting position
        ('stop_loss', 0.95),  # Stop loss as a percentage of the entry price
        ('hold_period', 10),  # Hold resistance/support for this number of bars
    )

    def __init__(self):
        self.highest_high = bt.indicators.Highest(self.data.high, period=self.params.lookback_period)
        self.lowest_low = bt.indicators.Lowest(self.data.low, period=self.params.exit_lookback_period)
        self.resistance = None  # Static resistance
        self.support = None  # Static support
        self.bar_count = 0  # To track when to update resistance/support
        self.order = None
        self.buy_signals = []  # List to store buy signal dates
        self.sell_signals = []  # List to store sell signal dates
        # self.buy_signals_count = 0  # Counter for buy signals
        # self.sell_signals_count = 0  # Counter for sell signals

    def next(self):
        # Log current prices and static resistance/support levels
        logging.info(f"Date: {self.data.datetime.date(0)}, Close: {self.data.close[0]:.2f}, "
                     f"Static Resistance: {self.resistance}, Static Support: {self.support}")

        # Update static resistance/support after holding for 'hold_period' bars
        if self.bar_count >= self.params.hold_period or self.resistance is None:
            self.resistance = self.highest_high[0]  # Freeze highest high
            self.support = self.lowest_low[0]  # Freeze lowest low
            self.bar_count = 0  # Reset bar count
        self.bar_count += 1

        # Exit if we have an open order
        if self.order:
            return

        # # Buy signal: If price breaks above static resistance
        # if not self.position:
        #     if self.data.close[0] > self.resistance:
        #         self.order = self.buy()
        #         self.stop_loss_price = self.data.close[0] * self.params.stop_loss
        #         self.buy_signals_count += 1  # Increment buy signal counter
        #         logging.info(f"BUY ORDER CREATED at {self.data.datetime.date(0)}, Price: {self.data.close[0]:.2f}, "
        #                      f"Stop Loss: {self.stop_loss_price:.2f}")

        # # Sell signal: If price breaks below static support or hits stop loss
        # elif self.data.close[0] < self.support or self.data.close[0] < self.stop_loss_price:
        #     self.order = self.sell()
        #     self.sell_signals_count += 1  # Increment sell signal counter
        #     logging.info(f"SELL ORDER CREATED at {self.data.datetime.date(0)}, Price: {self.data.close[0]:.2f}")
        if not self.position:
            if self.data.close[0] > self.resistance:
                self.order = self.buy()
                self.stop_loss_price = self.data.close[0] * self.params.stop_loss
                self.buy_signals.append(self.data.datetime.date(0))  # Store buy signal date
                logging.info(f"BUY SIGNAL at {self.data.datetime.date(0)}, Price: {self.data.close[0]:.2f}")

        elif self.data.close[0] < self.support or self.data.close[0] < self.stop_loss_price:
            self.order = self.sell()
            self.sell_signals.append(self.data.datetime.date(0))  # Store sell signal date
            logging.info(f"SELL SIGNAL at {self.data.datetime.date(0)}, Price: {self.data.close[0]:.2f}")

    def get_signals(self):
        return {
            'buy_signals': self.buy_signals,
            'sell_signals': self.sell_signals
        }

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            if order.isbuy():
                logging.info(f"BUY EXECUTED at {order.executed.price:.2f}, "
                             f"Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}")
            elif order.issell():
                logging.info(f"SELL EXECUTED at {order.executed.price:.2f}, "
                             f"Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}")

        # Reset the order and bar count
        self.order = None

    def stop(self):
        # Log the final count of buy and sell signals
        logging.info(f"Final Buy Signals Count: {self.buy_signals_count}")
        logging.info(f"Final Sell Signals Count: {self.sell_signals_count}")
