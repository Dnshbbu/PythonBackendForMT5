import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Set default renderer to the notebook or browser (adjust based on environment)
pio.renderers.default = "browser"

# Download historical data
def get_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock['Adj Close']

# Define tickers
sp500 = "^GSPC"    # S&P 500 Index
csi300 = "000300.SS"  # CSI 300 Index

# Set date range
start_date = "2010-01-01"
end_date = "2023-01-01"

# Get data
sp500_data = get_data(sp500, start_date, end_date)
csi300_data = get_data(csi300, start_date, end_date)

# Create a DataFrame to store both
data = pd.DataFrame({
    'S&P 500': sp500_data,
    'CSI 300': csi300_data
})

# Normalize the data for comparison (base = 100)
normalized_data = data / data.iloc[0] * 100

# Create plotly figure
fig = go.Figure()

# Add traces for S&P 500 and CSI 300
fig.add_trace(go.Scatter(x=normalized_data.index, y=normalized_data['S&P 500'],
                         mode='lines', name='S&P 500'))
fig.add_trace(go.Scatter(x=normalized_data.index, y=normalized_data['CSI 300'],
                         mode='lines', name='CSI 300'))

# Customize the layout
fig.update_layout(
    title='S&P 500 vs. CSI 300 Performance',
    xaxis_title='Date',
    yaxis_title='Normalized Price (Base = 100)',
    legend_title='Indices',
    template='plotly_dark',  # Optional: plotly_dark, plotly_white, etc.
    hovermode='x unified'    # Shows data from both lines on hover
)

# Show plot
fig.show()
