import backtrader as bt
import logging
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os

class SimpleCrossoverStrategy(bt.Strategy):
    """Simple Crossover Strategy: Buys and sells based on moving average crossovers."""
    
    params = (
        ('ticker', None),               # Ticker symbol
        ('fast_period', 50),            # Period for the fast moving average
        ('slow_period', 200),           # Period for the slow moving average
    )
    
    def __init__(self):
        self.order = None
        self.data_close = []
        self.buy_signals = []
        self.sell_signals = []
        
        # Initialize moving averages
        self.fast_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.slow_period)
        
        # Crossover indicator
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
    
    def next(self):
        self.data_close.append(self.data.close[0])
        
        # Check if an order is pending
        if self.order:
            return
        
        # If not in the market, look for a buy signal
        if not self.position:
            if self.crossover > 0:
                self.order = self.buy()
                self.buy_signals.append(len(self.data_close) - 1)
                logging.info(f"BUY ORDER CREATED at {self.data.datetime.date(0)}, Price: {self.data.close[0]}")
        else:
            # If in the market, look for a sell signal
            if self.crossover < 0:
                self.order = self.sell()
                self.sell_signals.append(len(self.data_close) - 1)
                logging.info(f"SELL ORDER CREATED at {self.data.datetime.date(0)}, Price: {self.data.close[0]}")
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return  # Order is still pending
        
        if order.status == order.Completed:
            if order.isbuy():
                logging.info(f"BUY EXECUTED at {order.executed.price:.2f}, "
                             f"Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")
            elif order.issell():
                logging.info(f"SELL EXECUTED at {order.executed.price:.2f}, "
                             f"Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logging.warning("Order Canceled/Margin/Rejected")
        
        self.order = None  # Reset order
    
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
        fig.add_trace(go.Scatter(
            y=self.data_close, 
            mode='lines', 
            name='Close Price',
            line=dict(color='blue')
        ))
        
        # Plot Fast and Slow Moving Averages
        fig.add_trace(go.Scatter(
            y=self.fast_ma, 
            mode='lines', 
            name=f'Fast MA ({self.params.fast_period})',
            line=dict(color='orange')
        ))
        fig.add_trace(go.Scatter(
            y=self.slow_ma, 
            mode='lines', 
            name=f'Slow MA ({self.params.slow_period})',
            line=dict(color='purple')
        ))
        
        # Plot buy signals
        for buy in self.buy_signals:
            fig.add_trace(go.Scatter(
                x=[buy],
                y=[self.data_close[buy]],
                mode='markers',
                marker=dict(color='green', symbol='triangle-up', size=10),
                name='Buy Signal' if buy == self.buy_signals[0] else None  # Avoid duplicate legend entries
            ))
        
        # Plot sell signals
        for sell in self.sell_signals:
            fig.add_trace(go.Scatter(
                x=[sell],
                y=[self.data_close[sell]],
                mode='markers',
                marker=dict(color='red', symbol='triangle-down', size=10),
                name='Sell Signal' if sell == self.sell_signals[0] else None  # Avoid duplicate legend entries
            ))
        
        # Update layout for the plot
        fig.update_layout(
            title=f'Backtest Result: {ticker} ({self.__class__.__name__})',
            xaxis_title='Time',
            yaxis_title='Price',
            legend=dict(orientation="h"),
            hovermode='x unified'
        )
        
        # Save the chart as an interactive HTML
        pio.write_html(fig, file=html_file_name, auto_open=False)
        logging.info(f"Interactive plot saved as {html_file_name}")
        
        # Store the chart path
        self.chart_path = html_file_name