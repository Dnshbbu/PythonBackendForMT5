# strategies/sma_crossover_with_rsi.py

import backtrader as bt
import logging
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os

class SMA_CrossOver_with_Indicators(bt.Strategy):
    """SMA Crossover Strategy with RSI Confirmation and Integrated Plotting."""
    
    params = (
        ('short_window', 20),
        ('long_window', 50),
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('ticker', None),  # Add ticker as a parameter
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

        # Store data for plotting
        self.data_close = []
        self.data_short_sma = []
        self.data_long_sma = []
        self.data_rsi = []
        self.buy_signals = []
        self.sell_signals = []

    def next(self):
        # Store data for plotting
        self.data_close.append(self.data.close[0])
        self.data_short_sma.append(self.short_sma[0])
        self.data_long_sma.append(self.long_sma[0])
        self.data_rsi.append(self.rsi[0])

        if self.order:
            return  # Wait for the order to complete

        # Debug: Print current indicator values
        logging.info(f'Date: {self.data.datetime.date(0)}, Short SMA: {self.short_sma[0]:.2f}, '
                     f'Long SMA: {self.long_sma[0]:.2f}, RSI: {self.rsi[0]:.2f}')

        # Buy Signal: Crossover occurs and RSI is below overbought threshold
        if not self.position:
            if self.crossover > 0 and self.rsi < self.params.rsi_overbought:
                self.order = self.buy()
                self.buy_signals.append(len(self.data_close) - 1)  # Store buy signal index
                logging.info(f"BUY ORDER CREATED at {self.data.datetime.date(0)}, "
                             f"Price: {self.data.close[0]}, RSI: {self.rsi[0]:.2f}")

        # Sell Signal: Crossover occurs and RSI is above oversold threshold
        elif self.crossover < 0 and self.rsi > self.params.rsi_oversold:
            self.order = self.sell()
            self.sell_signals.append(len(self.data_close) - 1)  # Store sell signal index
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

    def stop(self):
        # This method is called when the backtest is finished
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
        
        # Initialize Plotly figure with subplots
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Plot Close Price
        fig.add_trace(go.Scatter(y=self.data_close, mode='lines', name='Close Price'), row=1, col=1)

        # Plot Short SMA
        fig.add_trace(go.Scatter(y=self.data_short_sma, mode='lines', name='Short SMA'), row=1, col=1)

        # Plot Long SMA
        fig.add_trace(go.Scatter(y=self.data_long_sma, mode='lines', name='Long SMA'), row=1, col=1)

        # Plot RSI
        fig.add_trace(go.Scatter(y=self.data_rsi, mode='lines', name='RSI'), row=2, col=1)

        # Add RSI Overbought and Oversold lines
        fig.add_hline(y=self.params.rsi_overbought, line=dict(color='red', dash='dash'), row=2, col=1)
        fig.add_hline(y=self.params.rsi_oversold, line=dict(color='green', dash='dash'), row=2, col=1)

        # Plot buy signals
        fig.add_trace(go.Scatter(
            x=self.buy_signals,
            y=[self.data_close[i] for i in self.buy_signals],
            mode='markers',
            marker=dict(color='green', symbol='triangle-up', size=10),
            name='Buy Signal'
        ), row=1, col=1)

        # Plot sell signals
        fig.add_trace(go.Scatter(
            x=self.sell_signals,
            y=[self.data_close[i] for i in self.sell_signals],
            mode='markers',
            marker=dict(color='red', symbol='triangle-down', size=10),
            name='Sell Signal'
        ), row=1, col=1)

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