# strategies/breakout_bull_run.py

import backtrader as bt
import logging
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os

class BreakoutBullRun(bt.Strategy):
    params = (
        ('lookback_period', 10),  # Look back to find resistance/support
        ('exit_lookback_period', 5),  # Look back for exiting position
        ('stop_loss', 0.95),  # Stop loss as a percentage of the entry price
        ('hold_period', 10),  # Hold resistance/support for this number of bars
        ('ticker', None),  # Add ticker as a parameter
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

        # For plotting
        self.data_close = []
        self.data_high = []
        self.data_low = []
        self.resistance_line = []
        self.support_line = []

    def next(self):
        # Store data for plotting
        self.data_close.append(self.data.close[0])
        self.data_high.append(self.data.high[0])
        self.data_low.append(self.data.low[0])

        # Log current prices and static resistance/support levels
        logging.info(f"Date: {self.data.datetime.date(0)}, Close: {self.data.close[0]:.2f}, "
                     f"Static Resistance: {self.resistance}, Static Support: {self.support}")

        # Update static resistance/support after holding for 'hold_period' bars
        if self.bar_count >= self.params.hold_period or self.resistance is None:
            self.resistance = self.highest_high[0]  # Freeze highest high
            self.support = self.lowest_low[0]  # Freeze lowest low
            self.bar_count = 0  # Reset bar count
        self.bar_count += 1

        self.resistance_line.append(self.resistance)
        self.support_line.append(self.support)

        # Exit if we have an open order
        if self.order:
            return

        if not self.position:
            if self.data.close[0] > self.resistance:
                self.order = self.buy()
                self.stop_loss_price = self.data.close[0] * self.params.stop_loss
                self.buy_signals.append(len(self.data_close) - 1)  # Store buy signal index
                logging.info(f"BUY SIGNAL at {self.data.datetime.date(0)}, Price: {self.data.close[0]:.2f}")

        elif self.data.close[0] < self.support or self.data.close[0] < self.stop_loss_price:
            self.order = self.sell()
            self.sell_signals.append(len(self.data_close) - 1)  # Store sell signal index
            logging.info(f"SELL SIGNAL at {self.data.datetime.date(0)}, Price: {self.data.close[0]:.2f}")

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
        self.plot_results()

    def plot_results(self):
        # Create the 'plots' directory if it doesn't exist
        output_dir = "plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Set the HTML file name with folder
        #ticker = self.data._name
        # Use the ticker from the strategy parameters
        ticker = self.params.ticker if self.params.ticker else "Unknown"

        html_file_name = os.path.join(output_dir, f"{ticker}_{self.__class__.__name__}.html")
        
        logging.info(f"Creating interactive plot for strategy {self.__class__.__name__}: {html_file_name}")
        
        # Initialize Plotly figure
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
        
        # Plot Close Price
        fig.add_trace(go.Scatter(y=self.data_close, mode='lines', name='Close Price'))

        # Plot Resistance and Support lines
        fig.add_trace(go.Scatter(y=self.resistance_line, mode='lines', name='Resistance', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(y=self.support_line, mode='lines', name='Support', line=dict(color='green', dash='dash')))

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