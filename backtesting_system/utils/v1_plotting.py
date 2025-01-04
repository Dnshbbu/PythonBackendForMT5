
import os
import logging
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
from .rsi_calculation import calculate_rsi
import pandas as pd

def create_interactive_plot(stock_data, ticker, strategy_params, strategy_name):
    """
    Creates and saves an interactive Plotly chart for the backtest results.
    
    Parameters:
    - stock_data (pd.DataFrame): DataFrame containing stock data with 'Close' prices.
    - ticker (str): Stock ticker symbol.
    - strategy_params (dict): Dictionary of strategy parameters.
    - strategy_name (str): Name of the strategy being used.
    
    Returns:
    - str: Filename of the saved HTML plot.
    """

    # Create the 'plots' directory if it doesn't exist
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set the HTML file name with folder
    html_file_name = os.path.join(output_dir, f"{ticker}_{strategy_name}.html")
    
    logging.info(f"Creating interactive plot for strategy {strategy_name}: {html_file_name}")
    
    # Initialize Plotly figure with subplots
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    if strategy_name == 'BreakoutBullRun':
        lookback_period = strategy_params.get('lookback_period', 20)
        exit_lookback_period = strategy_params.get('exit_lookback_period', 10)
        stop_loss = strategy_params.get('stop_loss', 0.95)

        # Calculate highest highs and lowest lows for breakout and exit signals
        highest_high = stock_data['High'].rolling(window=lookback_period).max()
        lowest_low = stock_data['Low'].rolling(window=exit_lookback_period).min()

        stock_data['Highest_High'] = highest_high
        stock_data['Lowest_Low'] = lowest_low

        # Plot Close Price
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))

        # Plot Highest High
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Highest_High'], mode='lines', name=f'{lookback_period}-Day High'))

        # Plot Lowest Low
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Lowest_Low'], mode='lines', name=f'{exit_lookback_period}-Day Low'))

        # Identify breakout buy/sell signals
        buy_signals = stock_data['Close'] > stock_data['Highest_High']
        sell_signals = stock_data['Close'] < stock_data['Lowest_Low']

        logging.info(f"Buy signals count: {buy_signals.sum()}, Sell signals count: {sell_signals.sum()}")

        # Plot buy signals
        if buy_signals.any():
            fig.add_trace(go.Scatter(
                x=stock_data.index[buy_signals],
                y=stock_data['Close'][buy_signals],
                mode='markers',
                marker=dict(color='green', symbol='triangle-up', size=10),
                name='Buy Signal'
            ))

        # Plot sell signals
        if sell_signals.any():
            fig.add_trace(go.Scatter(
                x=stock_data.index[sell_signals],
                y=stock_data['Close'][sell_signals],
                mode='markers',
                marker=dict(color='red', symbol='triangle-down', size=10),
                name='Sell Signal'
            ))
    
    # Update layout for the plot
    fig.update_layout(
        title=f'Backtest Result: {ticker} ({strategy_name})',
        xaxis_title='Date',
        yaxis_title='Price',
        showlegend=True
    )
    
    # Save the chart as an interactive HTML under 'plots/' folder
    pio.write_html(fig, file=html_file_name, auto_open=False)
    logging.info(f"Interactive plot saved as {html_file_name}")
    
    return html_file_name