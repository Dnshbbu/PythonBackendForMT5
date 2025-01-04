import backtrader as bt
import logging
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os

class BuyAndHold(bt.Strategy):
    """Buy and Hold Strategy: Buys once and holds the position."""
    
    params = (
        ('ticker', None),  # Add ticker as a parameter
    )
    
    def __init__(self):
        self.order = None
        self.data_close = []
        self.buy_signal = None

    def next(self):
        self.data_close.append(self.data.close[0])
        
        if not self.position:
            self.order = self.buy()
            self.buy_signal = len(self.data_close) - 1
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

    def stop(self):
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
        
        # Initialize Plotly figure
        fig = go.Figure()
        
        # Plot Close Price
        fig.add_trace(go.Scatter(y=self.data_close, mode='lines', name='Close Price'))

        # Plot buy signal
        if self.buy_signal is not None:
            fig.add_trace(go.Scatter(
                x=[self.buy_signal],
                y=[self.data_close[self.buy_signal]],
                mode='markers',
                marker=dict(color='green', symbol='triangle-up', size=10),
                name='Buy Signal'
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