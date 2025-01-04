# strategies/buy_and_hold.py

import backtrader as bt
import logging

class BuyAndHold(bt.Strategy):
    """Buy and Hold Strategy: Buys once and holds the position."""
    
    def __init__(self):
        self.order = None

    def next(self):
        if not self.position:
            self.order = self.buy()
            logging.info(f"BUY ORDER CREATED at {self.data.datetime.date(0)}, Price: {self.data.close[0]}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                logging.info(f"BUY EXECUTED at {order.executed.price:.2f}, "
                             f"Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")
            elif order.issell():
                logging.info(f"SELL EXECUTED at {order.executed.price:.2f}, "
                             f"Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logging.warning("Order Canceled/Margin/Rejected")

        self.order = None
