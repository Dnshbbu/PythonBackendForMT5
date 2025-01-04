import pandas as pd
import plotly.graph_objects as go

# Load equity and price data from CSV file
equity_price_file_path = r'C:\Users\StdUser\Desktop\MyProjects\Backtesting\logs\INTC_10015335_priceandequity.csv'  # Replace with your CSV file path
df_equity_price = pd.read_csv(equity_price_file_path)

# Combine Date and Time into a single datetime column for equity and price data
df_equity_price['DateTime'] = pd.to_datetime(df_equity_price['Date'] + ' ' + df_equity_price['Time'])

# Load transaction data from CSV file
transaction_file_path = r'C:\Users\StdUser\Desktop\MyProjects\Backtesting\logs\INTC_10015335_deals_data.csv'  # Replace with your transaction CSV file path

# Read the transaction CSV file without headers and assign column names
df_transactions = pd.read_csv(
    transaction_file_path,
    header=None,
    names=['time', 'deal', 'symbol', 'type', 'direction', 'volume', 'price', 'order',
           'commission', 'swap', 'profit', 'balance', 'comment'],
    delimiter='\t',  # Adjust delimiter if needed
    engine='python'  # Use Python engine to handle irregular delimiters
)

# Create 'DateTime' column from 'time' column
df_transactions['DateTime'] = pd.to_datetime(df_transactions['time'], format='%Y.%m.%d %H:%M:%S')

# Plotting with Plotly
fig = go.Figure()

# Add trace for Price
fig.add_trace(go.Scatter(
    x=df_equity_price['DateTime'],
    y=df_equity_price['Price'],
    mode='lines',
    name='Price',
    yaxis='y1',
    hovertemplate='<b>Price</b><br>Date: %{x}<br>Price: %{y}<extra></extra>'
))

# Add trace for Equity
fig.add_trace(go.Scatter(
    x=df_equity_price['DateTime'],
    y=df_equity_price['Equity'],
    mode='lines',
    name='Equity',
    yaxis='y2',
    hovertemplate='<b>Equity</b><br>Date: %{x}<br>Equity: %{y}<extra></extra>'
))

# Filter Buy and Sell transactions
df_buy = df_transactions[(df_transactions['type'] == 'buy') & (df_transactions['direction'] == 'in')]
df_sell = df_transactions[(df_transactions['type'] == 'sell') & (df_transactions['direction'] == 'out')]

# Add Buy signals (Green upward triangle)
fig.add_trace(go.Scatter(
    x=df_buy['DateTime'],
    y=df_buy['price'],
    mode='markers',
    marker=dict(symbol='triangle-up', color='green', size=10),
    name='Buy',
    hovertemplate=(
        '<b>Buy</b><br>Date: %{x}<br>Price: %{y}<br>'
        'Volume: %{customdata[0]}<br>Profit: %{customdata[1]}<br>Balance: %{customdata[2]}<br>'
        'Comment: %{customdata[3]}<extra></extra>'
    ),
    customdata=df_buy[['volume', 'profit', 'balance', 'comment']].values
))

# Add Sell signals (Red inverted triangle)
fig.add_trace(go.Scatter(
    x=df_sell['DateTime'],
    y=df_sell['price'],
    mode='markers',
    marker=dict(symbol='triangle-down', color='red', size=10),
    name='Sell',
    hovertemplate=(
        '<b>Sell</b><br>Date: %{x}<br>Price: %{y}<br>'
        'Volume: %{customdata[0]}<br>Profit: %{customdata[1]}<br>Balance: %{customdata[2]}<br>'
        'Comment: %{customdata[3]}<extra></extra>'
    ),
    customdata=df_sell[['volume', 'profit', 'balance', 'comment']].values
))

# Update layout to include a secondary y-axis
fig.update_layout(
    title='Equity and Price with Buy/Sell Signals Over Time',
    xaxis_title='DateTime',
    yaxis_title='Price',
    yaxis2=dict(title='Equity', overlaying='y', side='right'),
    xaxis_rangeslider_visible=True,
    hovermode='x unified'
)

# Set x-axis range based on your data
min_date = df_equity_price['DateTime'].min()
max_date = df_equity_price['DateTime'].max()
fig.update_xaxes(range=[min_date, max_date])

# Save the figure to an HTML file
output_file_path = 'output_plot_with_signals.html'  # Specify the output HTML file name
fig.write_html(output_file_path)

print(f"Plot saved as {output_file_path}")
