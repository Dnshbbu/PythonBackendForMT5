import backtrader as bt
import logging
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
import pandas as pd

class SMACrossOver(bt.Strategy):
    """Simple Moving Average Crossover Strategy."""
    
    params = (
        ('ticker', None),          # Ticker symbol
        ('short_window', 50),      # Short SMA period
        ('long_window', 200),      # Long SMA period
    )

    def __init__(self):
        # Initialize Simple Moving Averages
        self.short_sma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.short_window, plotname='Short SMA'
        )
        self.long_sma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.long_window, plotname='Long SMA'
        )
        # Initialize Crossover Indicator
        self.crossover = bt.indicators.CrossOver(self.short_sma, self.long_sma)
        
        self.order = None
        self.data_close = []
        self.buy_signals = []   # Indices for buy signals
        self.sell_signals = []  # Indices for sell signals

        # Store dates for plotting
        self.dates = []

    def next(self):
        # Append current close price and date
        self.data_close.append(self.data.close[0])
        current_date = self.data.datetime.date(0)
        self.dates.append(current_date)

        if self.order:
            # If an order is pending, do nothing
            return

        logging.info(
            f'Date: {current_date}, Short SMA: {self.short_sma[0]:.2f}, '
            f'Long SMA: {self.long_sma[0]:.2f}'
        )

        # Check for crossover to generate buy/sell signals
        if self.crossover > 0 and not self.position:
            self.order = self.buy()
            self.buy_signals.append(len(self.data_close) - 1)  # Record buy index
            logging.info(f"BUY ORDER CREATED at {current_date}, Price: {self.data.close[0]:.2f}")

        elif self.crossover < 0 and self.position:
            self.order = self.sell()
            self.sell_signals.append(len(self.data_close) - 1)  # Record sell index
            logging.info(f"SELL ORDER CREATED at {current_date}, Price: {self.data.close[0]:.2f}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Order is submitted/accepted but not yet completed
            return

        if order.status == order.Completed:
            if order.isbuy():
                logging.info(
                    f"BUY EXECUTED at {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}"
                )
            elif order.issell():
                logging.info(
                    f"SELL EXECUTED at {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}"
                )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logging.warning("Order Canceled/Margin/Rejected")

        # Reset the order
        self.order = None

    def stop(self):
        # Plot results at the end of the backtest
        self.plot_results()

    def plot_results(self):
        # Create the 'plots' directory if it doesn't exist
        output_dir = "plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Use the ticker from the strategy parameters
        ticker = self.params.ticker if self.params.ticker else "Unknown"

        # Set the HTML file name with folder and ticker
        html_file_name = os.path.join(output_dir, f"{ticker}_{self.__class__.__name__}.html")
        
        logging.info(f"Creating interactive plot for strategy {self.__class__.__name__}: {html_file_name}")
        
        # Initialize Plotly figure with subplots if needed
        fig = go.Figure()

        # Ensure that dates and data_close have the same length
        if len(self.dates) != len(self.data_close):
            logging.warning("Length of dates and data_close do not match. Truncating to the shortest length.")
            min_length = min(len(self.dates), len(self.data_close))
            dates = self.dates[:min_length]
            data_close = self.data_close[:min_length]
            short_sma = list(self.short_sma.array)[:min_length]
            long_sma = list(self.long_sma.array)[:min_length]
        else:
            dates = self.dates
            data_close = self.data_close
            short_sma = list(self.short_sma.array)
            long_sma = list(self.long_sma.array)

        # Plot Close Price
        fig.add_trace(go.Scatter(
            x=dates, y=data_close, mode='lines', name='Close Price',
            line=dict(color='blue')
        ))

        # Plot Short SMA
        fig.add_trace(go.Scatter(
            x=dates, y=short_sma, mode='lines', name=f'Short SMA ({self.params.short_window})',
            line=dict(color='orange')
        ))

        # Plot Long SMA
        fig.add_trace(go.Scatter(
            x=dates, y=long_sma, mode='lines', name=f'Long SMA ({self.params.long_window})',
            line=dict(color='purple')
        ))

        # Plot Buy Signals
        if self.buy_signals:
            buy_dates = [dates[i] for i in self.buy_signals]
            buy_prices = [data_close[i] for i in self.buy_signals]
            fig.add_trace(go.Scatter(
                x=buy_dates, y=buy_prices,
                mode='markers',
                marker=dict(color='green', symbol='triangle-up', size=10),
                name='Buy Signal'
            ))

        # Plot Sell Signals
        if self.sell_signals:
            sell_dates = [dates[i] for i in self.sell_signals]
            sell_prices = [data_close[i] for i in self.sell_signals]
            fig.add_trace(go.Scatter(
                x=sell_dates, y=sell_prices,
                mode='markers',
                marker=dict(color='red', symbol='triangle-down', size=10),
                name='Sell Signal'
            ))

        # Update layout for the plot
        fig.update_layout(
            title=f'Backtest Result: {ticker} ({self.__class__.__name__})',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_dark',  # Optional: choose a theme
            legend=dict(
                x=0,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0)',
                bordercolor='rgba(255, 255, 255, 0)'
            )
        )
        
        # Save the chart as an interactive HTML
        pio.write_html(fig, file=html_file_name, auto_open=False)
        logging.info(f"Interactive plot saved as {html_file_name}")
        
        # Store the chart path
        self.chart_path = html_file_name
