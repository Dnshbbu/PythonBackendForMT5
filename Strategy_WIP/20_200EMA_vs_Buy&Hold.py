import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Parameters
TICKER = 'PYPL'          # Stock ticker symbol
START_DATE = '2010-01-01'  # Start date for historical data
END_DATE = '2023-10-31'    # End date for historical data

SHORT_EMA = 20           # Short-term EMA period
LONG_EMA = 200           # Long-term EMA period

INITIAL_CAPITAL = 100000  # Starting capital in USD

# Step 1: Fetch Historical Data
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    return df

# Step 2: Calculate EMAs
def calculate_emas(df, short_period, long_period):
    df['EMA_Short'] = df['Close'].ewm(span=short_period, adjust=False).mean()
    df['EMA_Long'] = df['Close'].ewm(span=long_period, adjust=False).mean()
    return df

# Step 3: Generate Buy/Sell Signals
def generate_signals(df, short_period, long_period):
    df['Signal'] = 0
    df['Signal'][short_period:] = np.where(
        df['EMA_Short'][short_period:] > df['EMA_Long'][short_period:], 1, 0
    )
    df['Position'] = df['Signal'].diff()
    return df

# Step 4: Backtest Strategy
def backtest_strategy(df, initial_capital):
    positions = pd.DataFrame(index=df.index).fillna(0.0)
    positions['Position'] = df['Signal']  # 1 for holding the stock, 0 for not

    # Calculate daily returns
    df['Market Return'] = df['Close'].pct_change()
    df['Strategy Return'] = df['Market Return'] * positions['Position'].shift(1)

    # Calculate cumulative returns
    df['Cumulative Market Return'] = (1 + df['Market Return']).cumprod() * initial_capital
    df['Cumulative Strategy Return'] = (1 + df['Strategy Return']).cumprod() * initial_capital

    return df

# Step 5: Plot Interactive Chart
def plot_results(df, ticker, short_period, long_period):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=(f'{ticker} Price with Buy/Sell Signals', 'Strategy vs Buy and Hold Performance'),
                        row_width=[0.2, 0.7])

    # Price chart with EMAs and buy/sell signals
    fig.add_trace(
        go.Candlestick(x=df.index,
                       open=df['Open'],
                       high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       name='Candlestick'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA_Short'], line=dict(color='blue', width=1), name=f'{short_period}-Day EMA'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA_Long'], line=dict(color='red', width=1), name=f'{long_period}-Day EMA'),
        row=1, col=1
    )

    # Buy signals
    buy_signals = df[df['Position'] == 1]
    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Close'],
            mode='markers',
            marker=dict(symbol='triangle-up', color='green', size=10),
            name='Buy Signal'
        ),
        row=1, col=1
    )

    # Sell signals
    sell_signals = df[df['Position'] == -1]
    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=sell_signals['Close'],
            mode='markers',
            marker=dict(symbol='triangle-down', color='red', size=10),
            name='Sell Signal'
        ),
        row=1, col=1
    )

    # Performance chart
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Cumulative Strategy Return'],
            line=dict(color='blue', width=2),
            name='Strategy Return'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Cumulative Market Return'],
            line=dict(color='orange', width=2),
            name='Buy and Hold Return'
        ),
        row=2, col=1
    )

    fig.update_layout(
        title=f'Backtesting Strategy: {short_period}-Day EMA Crossing {long_period}-Day EMA',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        legend=dict(x=0, y=1.2, orientation='h'),
        hovermode='x unified',
        height=800
    )

    fig.show()

# Step 6: Main Execution
if __name__ == "__main__":
    # Fetch data
    print("Fetching historical data...")
    df = fetch_data(TICKER, START_DATE, END_DATE)
    print("Data fetched successfully.")

    # Calculate EMAs
    print("Calculating EMAs...")
    df = calculate_emas(df, SHORT_EMA, LONG_EMA)
    print("EMAs calculated.")

    # Generate signals
    print("Generating buy/sell signals...")
    df = generate_signals(df, SHORT_EMA, LONG_EMA)
    print("Signals generated.")

    # Backtest strategy
    print("Backtesting the strategy...")
    df = backtest_strategy(df, INITIAL_CAPITAL)
    print("Backtesting completed.")

    # Display final portfolio values
    final_strategy = df['Cumulative Strategy Return'].iloc[-1]
    final_buy_hold = df['Cumulative Market Return'].iloc[-1]
    print(f"\nFinal Strategy Value: ${final_strategy:,.2f}")
    print(f"Final Buy and Hold Value: ${final_buy_hold:,.2f}")

    # Plot results
    print("Plotting the results...")
    plot_results(df, TICKER, SHORT_EMA, LONG_EMA)
