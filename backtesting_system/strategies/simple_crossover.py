# strategies/simple_crossover.py

import backtrader as bt
import logging
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os

class SimpleCrossOver(bt.Strategy):
    params = (
        ('fast_period', 10),        # Period for the fast moving average
        ('slow_period', 30),        # Period for the slow moving average
        ('stop_loss', 0.95),        # Stop loss as a percentage of the entry price
        ('ticker', None),           # Ticker symbol
    )

    # Define optimization parameters
    opt_params = {
        'fast_period': range(5, 21, 5),        # 5, 10, 15, 20
        'slow_period': range(20, 51, 10),      # 20, 30, 40, 50
        'stop_loss': [0.95, 0.90, 0.85],       # 5%, 10%, 15% stop loss
    }

    def __init__(self):
        # Initialize moving averages
        self.fast_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.slow_period)
        
        # Track crossovers
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        
        # Order tracking
        self.order = None
        
        # Lists to store signals and data for plotting
        self.buy_signals = []
        self.sell_signals = []
        self.data_close = []
        self.fast_ma_values = []
        self.slow_ma_values = []

    def next(self):
        # Store data for plotting
        self.data_close.append(self.data.close[0])
        self.fast_ma_values.append(self.fast_ma[0])
        self.slow_ma_values.append(self.slow_ma[0])
        
        # Log current state
        logging.info(f"Date: {self.data.datetime.date(0)}, Close: {self.data.close[0]:.2f}, "
                     f"Fast MA: {self.fast_ma[0]:.2f}, Slow MA: {self.slow_ma[0]:.2f}")
        
        # If an order is pending, do nothing
        if self.order:
            return
        
        # If not in the market, look for a buy signal
        if not self.position:
            if self.crossover > 0:
                self.order = self.buy()
                self.stop_loss_price = self.data.close[0] * self.params.stop_loss
                self.buy_signals.append(len(self.data_close) - 1)
                logging.info(f"BUY SIGNAL at {self.data.datetime.date(0)}, Price: {self.data.close[0]:.2f}")
        
        # If in the market, look for a sell signal
        else:
            if self.crossover < 0 or self.data.close[0] < self.stop_loss_price:
                self.order = self.sell()
                self.sell_signals.append(len(self.data_close) - 1)
                logging.info(f"SELL SIGNAL at {self.data.datetime.date(0)}, Price: {self.data.close[0]:.2f}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Order is pending, do nothing
            return
        
        if order.status == order.Completed:
            if order.isbuy():
                logging.info(f"BUY EXECUTED at {order.executed.price:.2f}, "
                             f"Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}")
            elif order.issell():
                logging.info(f"SELL EXECUTED at {order.executed.price:.2f}, "
                             f"Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}")
        
        # Reset order
        self.order = None

    def stop(self):
        self.plot_results()

    def plot_results(self):
        # Create the 'plots' directory if it doesn't exist
        output_dir = "plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Use the ticker from the strategy parameters
        ticker = self.params.ticker if self.params.ticker else "Unknown"
        html_file_name = os.path.join(output_dir, f"{ticker}_{self.__class__.__name__}.html")
        
        logging.info(f"Creating interactive plot for strategy {self.__class__.__name__}: {html_file_name}")
        
        # Initialize Plotly figure
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
        
        # Plot Close Price
        fig.add_trace(go.Scatter(y=self.data_close, mode='lines', name='Close Price'))
        
        # Plot Fast and Slow Moving Averages
        fig.add_trace(go.Scatter(y=self.fast_ma_values, mode='lines', name=f'Fast MA ({self.params.fast_period})', line=dict(color='blue')))
        fig.add_trace(go.Scatter(y=self.slow_ma_values, mode='lines', name=f'Slow MA ({self.params.slow_period})', line=dict(color='orange')))
        
        # Plot buy signals
        fig.add_trace(go.Scatter(
            x=self.buy_signals,
            y=[self.data_close[i] for i in self.buy_signals],
            mode='markers',
            marker=dict(color='green', symbol='triangle-up', size=10),
            name='Buy Signal'
        ))
        
        # Plot sell signals
        fig.add_trace(go.Scatter(
            x=self.sell_signals,
            y=[self.data_close[i] for i in self.sell_signals],
            mode='markers',
            marker=dict(color='red', symbol='triangle-down', size=10),
            name='Sell Signal'
        ))
        
        # Update layout for the plot
        fig.update_layout(
            title=f'Backtest Result: {ticker} ({self.__class__.__name__})',
            xaxis_title='Date',
            yaxis_title='Price',
            showlegend=True
        )
        
        # Save the chart as an interactive HTML
        pio.write_html(fig, file=html_file_name, auto_open=False)
        logging.info(f"Interactive plot saved as {html_file_name}")
        
        # Store the chart path
        self.chart_path = html_file_name
