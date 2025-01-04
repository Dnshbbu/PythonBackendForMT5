import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from fredapi import Fred

# Fetch data from FRED and Yahoo Finance
def get_data():
    # Fetch interest rate data from FRED
    fred = Fred(api_key='c18fd978df29c2a606c588977da9823e')  # Replace 'YOUR_FRED_API_KEY' with your FRED API key
    interest_rate_data = fred.get_series('FEDFUNDS')  # Federal Funds Effective Rate (or use other codes)

    # Convert to DataFrame
    interest_rate_df = pd.DataFrame(interest_rate_data, columns=['Rate'])
    interest_rate_df.index.name = 'Date'
    interest_rate_df.reset_index(inplace=True)
    interest_rate_df['Date'] = pd.to_datetime(interest_rate_df['Date'])  # Ensure date is in datetime format

    # Fetch S&P 500 data from Yahoo Finance
    snp_data = yf.download('^GSPC', start='2000-01-01', end='2023-01-01')

    # Convert to DataFrame
    snp_df = pd.DataFrame(snp_data['Close'])
    snp_df.index.name = 'Date'
    snp_df.reset_index(inplace=True)
    snp_df['Date'] = pd.to_datetime(snp_df['Date'])  # Ensure date is in datetime format

    # Align data to the same date range
    start_date = max(interest_rate_df['Date'].min(), snp_df['Date'].min())
    end_date = min(interest_rate_df['Date'].max(), snp_df['Date'].max())
    
    interest_rate_df = interest_rate_df[(interest_rate_df['Date'] >= start_date) & (interest_rate_df['Date'] <= end_date)]
    snp_df = snp_df[(snp_df['Date'] >= start_date) & (snp_df['Date'] <= end_date)]

    return interest_rate_df, snp_df

# Plot interest rates and S&P 500 with dual y-axes
def plot_data(interest_rate_df, snp_df):
    fig = go.Figure()

    # Add Interest Rate as a line graph
    fig.add_trace(go.Scatter(
        x=interest_rate_df['Date'],
        y=interest_rate_df['Rate'],
        mode='lines',
        name='Interest Rate',
        line=dict(color='blue'),
        yaxis='y2'  # Use secondary y-axis
    ))

    # Add S&P 500 as a line graph
    fig.add_trace(go.Scatter(
        x=snp_df['Date'],
        y=snp_df['Close'],
        mode='lines',
        name='S&P 500',
        line=dict(color='red'),
    ))

    # Update layout to include dual y-axes
    fig.update_layout(
        title="Interest Rate and S&P 500",
        xaxis_title="Date",
        yaxis_title="S&P 500 Value",
        yaxis2=dict(
            title='Interest Rate (%)',
            overlaying='y',
            side='right'
        ),
        plot_bgcolor="white",
        font=dict(family="Arial", size=12),
        legend=dict(x=0.05, y=0.95),
        xaxis_rangeslider_visible=False  # Hide the range slider for simplicity
    )

    # Show the interactive plot
    fig.show()

if __name__ == '__main__':
    # Get data
    interest_rate_df, snp_df = get_data()

    # Plot data
    plot_data(interest_rate_df, snp_df)
