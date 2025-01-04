import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# Parameters
TICKERS = ['ANTX', 'MCRB', 'HUM', '^GSPC']  # Updated list of stock ticker symbols
START_DATE = '2010-01-01'
END_DATE = '2023-10-31'
SHORT_EMA = 20
LONG_EMA = 200
INITIAL_CAPITAL = 100000
RSI_PERIOD = 14
VOLUME_THRESHOLD = 1.5  # Volume must be 1.5 times the average for confirmation
TRAILING_STOP_LOSS = 0.05  # 5% trailing stop loss

# Function Definitions

def fetch_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            print(f"Warning: No data fetched for ticker '{ticker}'.")
            return None
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data for ticker '{ticker}': {e}")
        return None

def calculate_emas(df, short_period, long_period):
    df['EMA_Short'] = df['Close'].ewm(span=short_period, adjust=False).mean()
    df['EMA_Long'] = df['Close'].ewm(span=long_period, adjust=False).mean()
    return df

def calculate_rsi(df, period):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def generate_signals(df):
    df['Signal'] = 0
    df['Signal'].iloc[SHORT_EMA:] = np.where(
        (df['EMA_Short'].iloc[SHORT_EMA:] > df['EMA_Long'].iloc[SHORT_EMA:]) &
        (df['RSI'].iloc[SHORT_EMA:] > 50), 1, 0
    )
    df['Position'] = df['Signal'].diff()
    return df

def momentum_strategy(df):
    df['Breakout'] = df['Close'] > df['Close'].rolling(window=20).max().shift(1)
    df['Volume_Avg'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Confirmation'] = df['Volume'] > (VOLUME_THRESHOLD * df['Volume_Avg'])
    df['Entry'] = df['Breakout'] & df['Volume_Confirmation']

    # Initialize Stop_Loss column
    df['Stop_Loss'] = np.nan
    
    for i in range(1, len(df)):
        # Apply trailing stop logic
        if df['Entry'].iloc[i]:
            df['Stop_Loss'].iloc[i] = df['Close'].iloc[i] * (1 - TRAILING_STOP_LOSS)
        else:
            df['Stop_Loss'].iloc[i] = df['Stop_Loss'].iloc[i-1]  # Carry forward the last stop loss

    return df

def backtest_strategy(df):
    positions = pd.DataFrame(index=df.index).fillna(0.0)
    positions['Position'] = df['Signal']  # 1 for holding the stock, 0 for not
    
    # Calculate daily returns
    df['Market Return'] = df['Close'].pct_change()
    df['Strategy Return'] = df['Market Return'] * positions['Position'].shift(1)

    # Calculate cumulative returns
    df['Cumulative Market Return'] = (1 + df['Market Return']).cumprod() * INITIAL_CAPITAL
    df['Cumulative Strategy Return'] = (1 + df['Strategy Return']).cumprod() * INITIAL_CAPITAL

    return df

def plot_results(df, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=(f'{ticker} Price with Buy/Sell Signals',
                                        'Strategy vs Buy and Hold Performance'),
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
        go.Scatter(x=df.index, y=df['EMA_Short'], line=dict(color='blue', width=1), name=f'{SHORT_EMA}-Day EMA'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA_Long'], line=dict(color='red', width=1), name=f'{LONG_EMA}-Day EMA'),
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
        title=f'Backtesting Strategy for {ticker}',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        legend=dict(x=0, y=1.2, orientation='h'),
        hovermode='x unified',
        height=800
    )

    fig.show()

def main():
    final_values = {}
    
    for ticker in TICKERS:
        print(f"\nProcessing ticker: {ticker}")

        df = fetch_data(ticker, START_DATE, END_DATE)
        if df is None:
            continue

        df = calculate_emas(df, SHORT_EMA, LONG_EMA)
        df = calculate_rsi(df, RSI_PERIOD)
        df = generate_signals(df)
        df = momentum_strategy(df)
        df = backtest_strategy(df)

        # Store final values
        final_strategy = df['Cumulative Strategy Return'].iloc[-1]
        final_buy_hold = df['Cumulative Market Return'].iloc[-1]
        final_values[ticker] = {
            'Strategy': final_strategy,
            'Buy and Hold': final_buy_hold
        }

        # Plot results
        plot_results(df, ticker)

    print("\nFinal Portfolio Values:")
    print(pd.DataFrame(final_values).T)

if __name__ == "__main__":
    main()
