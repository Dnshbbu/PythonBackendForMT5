# utils/plotting.py

import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
from .rsi_calculation import calculate_rsi
import logging
import os

def create_interactive_plot(stock_data, ticker, strategy_params, strategy_name, signals=None):
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
    # html_file_name = f"{ticker}_{strategy_name}.html"

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
        # subplot_titles=(f'{ticker} Price with Indicators', 'RSI Indicator' if 'RSI' in stock_data else ''),
        row_heights=[0.7, 0.3]
    )
    
# Customize indicators based on strategy_name
    if strategy_name == 'SMA_CrossOver_with_Indicators':
        short_window = strategy_params.get('short_window', 20)
        long_window = strategy_params.get('long_window', 50)
        rsi_period = strategy_params.get('rsi_period', 14)
        rsi_overbought = strategy_params.get('rsi_overbought', 70)
        rsi_oversold = strategy_params.get('rsi_oversold', 30)

        # Calculate short and long SMAs and RSI
        short_sma = stock_data['Close'].rolling(window=short_window).mean()
        long_sma = stock_data['Close'].rolling(window=long_window).mean()
        rsi = calculate_rsi(stock_data['Close'], period=rsi_period)

        stock_data['Short_SMA'] = short_sma
        stock_data['Long_SMA'] = long_sma
        stock_data['RSI'] = rsi

        # Plot Close Price
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'), row=1, col=1)

        # Plot Short SMA
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Short_SMA'], mode='lines', name='Short SMA'), row=1, col=1)

        # Plot Long SMA
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Long_SMA'], mode='lines', name='Long SMA'), row=1, col=1)

        # Plot RSI
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'), row=2, col=1)

        # Add RSI Overbought and Oversold lines
        fig.add_hline(y=rsi_overbought, line=dict(color='red', dash='dash'), row=2, col=1)
        fig.add_hline(y=rsi_oversold, line=dict(color='green', dash='dash'), row=2, col=1)

        # Identify crossover buy/sell signals with RSI confirmation
        buy_signals = (short_sma > long_sma) & (short_sma.shift(1) <= long_sma.shift(1)) & (rsi < rsi_overbought)
        sell_signals = (short_sma < long_sma) & (short_sma.shift(1) >= long_sma.shift(1)) & (rsi > rsi_oversold)

        # Plot buy signals
        fig.add_trace(go.Scatter(
            x=stock_data.index[buy_signals],
            y=stock_data['Close'][buy_signals],
            mode='markers',
            marker=dict(color='green', symbol='triangle-up', size=10),
            name='Buy Signal'
        ), row=1, col=1)

        # Plot sell signals
        fig.add_trace(go.Scatter(
            x=stock_data.index[sell_signals],
            y=stock_data['Close'][sell_signals],
            mode='markers',
            marker=dict(color='red', symbol='triangle-down', size=10),
            name='Sell Signal'
        ), row=1, col=1)

    elif strategy_name == 'BuyAndHold':
        # Buy and Hold may only plot the close price
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'), row=1, col=1)

        # SMA_CrossOver strategy plotting logic
    
    elif strategy_name == 'SMACrossOver':
        short_window = strategy_params.get('short_window', 50)
        long_window = strategy_params.get('long_window', 200)

        # Calculate short and long SMAs
        short_sma = stock_data['Close'].rolling(window=short_window).mean()
        long_sma = stock_data['Close'].rolling(window=long_window).mean()

        stock_data['Short_SMA'] = short_sma
        stock_data['Long_SMA'] = long_sma

        # Plot Close Price
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'), row=1, col=1)

        # Plot Short SMA
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Short_SMA'], mode='lines', name='Short SMA'), row=1, col=1)

        # Plot Long SMA
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Long_SMA'], mode='lines', name='Long SMA'), row=1, col=1)

        # Plot crossover points (buy/sell signals)
        buy_signals = (short_sma > long_sma) & (short_sma.shift(1) <= long_sma.shift(1))
        sell_signals = (short_sma < long_sma) & (short_sma.shift(1) >= long_sma.shift(1))

        # Plot buy signals
        fig.add_trace(go.Scatter(
            x=stock_data.index[buy_signals],
            y=stock_data['Close'][buy_signals],
            mode='markers',
            marker=dict(color='green', symbol='triangle-up', size=10),
            name='Buy Signal'
        ), row=1, col=1)

        # Plot sell signals
        fig.add_trace(go.Scatter(
            x=stock_data.index[sell_signals],
            y=stock_data['Close'][sell_signals],
            mode='markers',
            marker=dict(color='red', symbol='triangle-down', size=10),
            name='Sell Signal'
        ), row=1, col=1)
    
    # elif strategy_name == 'BreakoutBullRun':
    #     lookback_period = strategy_params.get('lookback_period', 20)
    #     exit_lookback_period = strategy_params.get('exit_lookback_period', 10)
    #     stop_loss = strategy_params.get('stop_loss', 0.95)

    #     # Calculate highest highs and lowest lows for breakout and exit signals
    #     highest_high = stock_data['High'].rolling(window=lookback_period).max()
    #     lowest_low = stock_data['Low'].rolling(window=exit_lookback_period).min()

    #     stock_data['Highest_High'] = highest_high
    #     stock_data['Lowest_Low'] = lowest_low

    #     # Plot Close Price
    #     fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'), row=1, col=1)

    #     # Plot Highest High
    #     fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Highest_High'], mode='lines', name=f'{lookback_period}-Day High'), row=1, col=1)

    #     # Plot Lowest Low
    #     fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Lowest_Low'], mode='lines', name=f'{exit_lookback_period}-Day Low'), row=1, col=1)

    #     # Identify breakout buy/sell signals
    #     buy_signals = stock_data['Close'] > stock_data['Highest_High']
    #     sell_signals = stock_data['Close'] < stock_data['Lowest_Low']

    #     # Plot buy signals
    #     fig.add_trace(go.Scatter(
    #         x=stock_data.index[buy_signals],
    #         y=stock_data['Close'][buy_signals],
    #         mode='markers',
    #         marker=dict(color='green', symbol='triangle-up', size=10),
    #         name='Buy Signal'
    #     ), row=1, col=1)

    #     # Plot sell signals
    #     fig.add_trace(go.Scatter(
    #         x=stock_data.index[sell_signals],
    #         y=stock_data['Close'][sell_signals],
    #         mode='markers',
    #         marker=dict(color='red', symbol='triangle-down', size=10),
    #         name='Sell Signal'
    #     ), row=1, col=1)

    elif strategy_name == 'BreakoutBullRun':
        lookback_period = strategy_params.get('lookback_period', 10)
        exit_lookback_period = strategy_params.get('exit_lookback_period', 5)
        stop_loss = strategy_params.get('stop_loss', 0.95)

        highest_high = stock_data['High'].rolling(window=lookback_period).max()
        lowest_low = stock_data['Low'].rolling(window=exit_lookback_period).min()

        stock_data['Highest_High'] = highest_high
        stock_data['Lowest_Low'] = lowest_low

        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Highest_High'], mode='lines', name=f'{lookback_period}-Day High'))
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Lowest_Low'], mode='lines', name=f'{exit_lookback_period}-Day Low'))

        if signals:
            buy_signals = signals.get('buy_signals', [])
            sell_signals = signals.get('sell_signals', [])

            if buy_signals:
                fig.add_trace(go.Scatter(
                    x=buy_signals,
                    y=[stock_data.loc[date, 'Close'] for date in buy_signals],
                    mode='markers',
                    marker=dict(color='green', symbol='triangle-up', size=10),
                    name='Buy Signal'
                ))

            if sell_signals:
                fig.add_trace(go.Scatter(
                    x=sell_signals,
                    y=[stock_data.loc[date, 'Close'] for date in sell_signals],
                    mode='markers',
                    marker=dict(color='red', symbol='triangle-down', size=10),
                    name='Sell Signal'
                ))

        # Update layout for better readability
        # fig.update_layout(
        #     title='Stock Price with Buy and Sell Signals',
        #     xaxis_title='Date',
        #     yaxis_title='Price',
        #     legend_title='Legend',
        #     template='plotly_white'
        # )

        # Show the figure
        # fig.show()
    
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
