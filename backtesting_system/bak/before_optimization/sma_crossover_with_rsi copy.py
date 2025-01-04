# strategies/sma_crossover_with_rsi.py

import backtrader as bt
import logging

class SMA_CrossOver_with_Indicators(bt.Strategy):
    """SMA Crossover Strategy with RSI Confirmation."""
    
    params = (
        ('short_window', 20),
        ('long_window', 50),
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
    )

    def __init__(self):
        # Initialize SMA indicators
        self.short_sma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.short_window, plotname='Short SMA'
        )
        self.long_sma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.long_window, plotname='Long SMA'
        )

        # Initialize RSI indicator
        self.rsi = bt.indicators.RelativeStrengthIndex(
            self.data.close, period=self.params.rsi_period, plotname='RSI'
        )

        # Initialize Crossover indicator
        self.crossover = bt.indicators.CrossOver(self.short_sma, self.long_sma)

        # Track orders
        self.order = None

    def next(self):
        if self.order:
            return  # Wait for the order to complete

        # Debug: Print current indicator values
        logging.info(f'Date: {self.data.datetime.date(0)}, Short SMA: {self.short_sma[0]:.2f}, '
                     f'Long SMA: {self.long_sma[0]:.2f}, RSI: {self.rsi[0]:.2f}')

        # Buy Signal: Crossover occurs and RSI is below overbought threshold
        if not self.position:
            if self.crossover > 0 and self.rsi < self.params.rsi_overbought:
                self.order = self.buy()
                logging.info(f"BUY ORDER CREATED at {self.data.datetime.date(0)}, "
                             f"Price: {self.data.close[0]}, RSI: {self.rsi[0]:.2f}")

        # Sell Signal: Crossover occurs and RSI is above oversold threshold
        elif self.crossover < 0 and self.rsi > self.params.rsi_oversold:
            self.order = self.sell()
            logging.info(f"SELL ORDER CREATED at {self.data.datetime.date(0)}, "
                         f"Price: {self.data.close[0]}, RSI: {self.rsi[0]:.2f}")

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

        # Reset the order
        self.order = None
