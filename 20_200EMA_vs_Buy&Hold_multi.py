import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Suppress SettingWithCopyWarning and FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# Parameters
TICKERS = ['SYM', 'SNAP', 'HUM']  # List of stock ticker symbols
START_DATE = '2021-01-01'  # Start date for historical data
END_DATE = '2023-10-31'    # End date for historical data

SHORT_EMA = 20             # Short-term EMA period
LONG_EMA = 200             # Long-term EMA period

INITIAL_CAPITAL = 100000   # Starting capital in USD per symbol

# Function Definitions

def fetch_data(ticker, start, end):
    """
    Fetch historical stock data for a given ticker.
    """
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            print(f"Warning: No data fetched for ticker '{ticker}'. Please check the ticker symbol and date range.")
            return None
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data for ticker '{ticker}': {e}")
        return None

def calculate_emas(df, short_period, long_period):
    """
    Calculate short-term and long-term EMAs.
    """
    df['EMA_Short'] = df['Close'].ewm(span=short_period, adjust=False).mean()
    df['EMA_Long'] = df['Close'].ewm(span=long_period, adjust=False).mean()
    return df

def generate_signals(df, short_period, long_period):
    """
    Generate buy/sell signals based on EMA crossovers.
    """
    df['Signal'] = 0
    # Use iloc for position-based slicing
    df['Signal'].iloc[short_period:] = np.where(
        df['EMA_Short'].iloc[short_period:] > df['EMA_Long'].iloc[short_period:], 1, 0
    )
    df['Position'] = df['Signal'].diff()
    return df

def backtest_strategy(df, initial_capital):
    """
    Backtest the EMA crossover strategy.
    """
    positions = pd.DataFrame(index=df.index).fillna(0.0)
    positions['Position'] = df['Signal']  # 1 for holding the stock, 0 for not

    # Calculate daily returns
    df['Market Return'] = df['Close'].pct_change()
    df['Strategy Return'] = df['Market Return'] * positions['Position'].shift(1)

    # Calculate cumulative returns
    df['Cumulative Market Return'] = (1 + df['Market Return']).cumprod() * initial_capital
    df['Cumulative Strategy Return'] = (1 + df['Strategy Return']).cumprod() * initial_capital

    return df

def plot_results(df, ticker, short_period, long_period):
    """
    Plot interactive charts for a given ticker.
    """
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
        title=f'Backtesting Strategy: {short_period}-Day EMA Crossing {long_period}-Day EMA for {ticker}',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        legend=dict(x=0, y=1.2, orientation='h'),
        hovermode='x unified',
        height=800
    )

    fig.show()

# def main():
#     """
#     Main function to execute the backtesting strategy for multiple tickers.
#     """
#     # Initialize dictionaries to store results
#     final_values = {}
#     strategy_returns = {}
#     buy_hold_returns = {}

#     # Initialize a DataFrame to store combined portfolio returns
#     combined_strategy = None
#     combined_buy_hold = None

#     for ticker in TICKERS:
#         print(f"\nProcessing ticker: {ticker}")

#         # Step 1: Fetch data
#         df = fetch_data(ticker, START_DATE, END_DATE)
#         if df is None:
#             continue  # Skip to next ticker if data is not fetched

#         # Step 2: Calculate EMAs
#         df = calculate_emas(df, SHORT_EMA, LONG_EMA)

#         # Step 3: Generate signals
#         df = generate_signals(df, SHORT_EMA, LONG_EMA)

#         # Step 4: Backtest strategy
#         df = backtest_strategy(df, INITIAL_CAPITAL)

#         # Step 5: Store final values
#         final_strategy = df['Cumulative Strategy Return'].iloc[-1]
#         final_buy_hold = df['Cumulative Market Return'].iloc[-1]
#         final_values[ticker] = {
#             'Strategy': final_strategy,
#             'Buy and Hold': final_buy_hold
#         }
#         strategy_returns[ticker] = df['Strategy Return'].fillna(0)
#         buy_hold_returns[ticker] = df['Market Return'].fillna(0)

#         # Step 6: Plot results
#         plot_results(df, ticker, SHORT_EMA, LONG_EMA)

#     # Aggregate Profit/Loss
#     if strategy_returns and buy_hold_returns:
#         # Combine the returns by summing them (assuming equal allocation)
#         combined_strategy = pd.concat(strategy_returns.values(), axis=1).sum(axis=1)
#         combined_buy_hold = pd.concat(buy_hold_returns.values(), axis=1).sum(axis=1)

#         # Calculate cumulative returns
#         combined_initial_capital = INITIAL_CAPITAL * len(strategy_returns)
#         combined_cumulative_strategy = (1 + combined_strategy).cumprod() * combined_initial_capital
#         combined_cumulative_buy_hold = (1 + combined_buy_hold).cumprod() * combined_initial_capital

#         # Store combined final values
#         final_values['Combined'] = {
#             'Strategy': combined_cumulative_strategy.iloc[-1],
#             'Buy and Hold': combined_cumulative_buy_hold.iloc[-1]
#         }

#         # Create a summary DataFrame
#         summary_df = pd.DataFrame(final_values).T
#         summary_df = summary_df.rename_axis('Ticker')
#         summary_df = summary_df.reset_index()
#         summary_df = summary_df.rename(columns={'index': 'Ticker'})
#         print("\nFinal Portfolio Values:")
#         print(summary_df)

#         # Plot Combined Performance
#         fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
#                             subplot_titles=('Combined Strategy vs Buy and Hold Performance',),
#                             row_width=[0.7])

#         # Performance chart
#         fig.add_trace(
#             go.Scatter(
#                 x=combined_cumulative_strategy.index,
#                 y=combined_cumulative_strategy,
#                 line=dict(color='blue', width=2),
#                 name='Combined Strategy Return'
#             ),
#             row=1, col=1
#         )
#         fig.add_trace(
#             go.Scatter(
#                 x=combined_cumulative_buy_hold.index,
#                 y=combined_cumulative_buy_hold,
#                 line=dict(color='orange', width=2),
#                 name='Combined Buy and Hold Return'
#             ),
#             row=1, col=1
#         )

#         fig.update_layout(
#             title=f'Combined Backtesting Strategy: {SHORT_EMA}-Day EMA Crossing {LONG_EMA}-Day EMA',
#             yaxis_title='Cumulative Return (USD)',
#             xaxis_title='Date',
#             legend=dict(x=0, y=1.2, orientation='h'),
#             hovermode='x unified',
#             height=600
#         )

#         fig.show()

#         # Display the summary table
#         print("\nSummary of Final Portfolio Values:")
#         print(summary_df.to_string(index=False))

#     else:
#         print("\nNo valid tickers were processed. Please check the ticker symbols and data availability.")

# if __name__ == "__main__":
#     main()


def main():
    """
    Main function to execute the backtesting strategy for multiple tickers.
    """
    # Initialize dictionaries to store results
    final_values = {}
    strategy_returns = {}
    buy_hold_returns = {}

    # Initialize a DataFrame to store combined portfolio returns
    combined_strategy = None
    combined_buy_hold = None

    for ticker in TICKERS:
        print(f"\nProcessing ticker: {ticker}")

        # Step 1: Fetch data
        df = fetch_data(ticker, START_DATE, END_DATE)
        if df is None:
            continue  # Skip to next ticker if data is not fetched

        # Step 2: Calculate EMAs
        df = calculate_emas(df, SHORT_EMA, LONG_EMA)

        # Step 3: Generate signals
        df = generate_signals(df, SHORT_EMA, LONG_EMA)

        # Step 4: Backtest strategy
        df = backtest_strategy(df, INITIAL_CAPITAL)

        # Step 5: Store final values
        final_strategy = df['Cumulative Strategy Return'].iloc[-1]
        final_buy_hold = df['Cumulative Market Return'].iloc[-1]
        final_values[ticker] = {
            'Strategy': final_strategy,
            'Buy and Hold': final_buy_hold
        }
        strategy_returns[ticker] = df['Strategy Return'].fillna(0)
        buy_hold_returns[ticker] = df['Market Return'].fillna(0)

        # Step 6: Plot results
        plot_results(df, ticker, SHORT_EMA, LONG_EMA)

    # Aggregate Profit/Loss
    if strategy_returns and buy_hold_returns:
        # Combine the returns by summing them (assuming equal allocation)
        combined_strategy = pd.concat(strategy_returns.values(), axis=1).sum(axis=1)
        combined_buy_hold = pd.concat(buy_hold_returns.values(), axis=1).sum(axis=1)

        # Calculate cumulative returns
        combined_initial_capital = INITIAL_CAPITAL * len(strategy_returns)
        combined_cumulative_strategy = (1 + combined_strategy).cumprod() * combined_initial_capital
        combined_cumulative_buy_hold = (1 + combined_buy_hold).cumprod() * combined_initial_capital

        # Store combined final values
        final_values['Combined'] = {
            'Strategy': combined_cumulative_strategy.iloc[-1],
            'Buy and Hold': combined_cumulative_buy_hold.iloc[-1]
        }

        # Create a summary DataFrame
        summary_df = pd.DataFrame(final_values).T
        summary_df = summary_df.rename_axis('Ticker')
        summary_df = summary_df.reset_index()
        summary_df = summary_df.rename(columns={'index': 'Ticker'})
        print("\nFinal Portfolio Values:")
        # Modified Print Statement with Two Decimal Points
        print(summary_df.to_string(index=False, float_format='%.2f'))

        # Plot Combined Performance
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
                            subplot_titles=('Combined Strategy vs Buy and Hold Performance',),
                            row_width=[0.7])

        # Performance chart
        fig.add_trace(
            go.Scatter(
                x=combined_cumulative_strategy.index,
                y=combined_cumulative_strategy,
                line=dict(color='blue', width=2),
                name='Combined Strategy Return'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=combined_cumulative_buy_hold.index,
                y=combined_cumulative_buy_hold,
                line=dict(color='orange', width=2),
                name='Combined Buy and Hold Return'
            ),
            row=1, col=1
        )

        fig.update_layout(
            title=f'Combined Backtesting Strategy: {SHORT_EMA}-Day EMA Crossing {LONG_EMA}-Day EMA',
            yaxis_title='Cumulative Return (USD)',
            xaxis_title='Date',
            legend=dict(x=0, y=1.2, orientation='h'),
            hovermode='x unified',
            height=600
        )

        fig.show()

        # Display the summary table
        print("\nSummary of Final Portfolio Values:")
        print(summary_df.to_string(index=False, float_format='%.2f'))

    else:
        print("\nNo valid tickers were processed. Please check the ticker symbols and data availability.")

if __name__ == "__main__":
    main()