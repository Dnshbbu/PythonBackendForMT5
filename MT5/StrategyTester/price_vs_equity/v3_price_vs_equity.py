import pandas as pd
import plotly.graph_objects as go

# Load equity and price data from CSV file
equity_price_file_path = r'C:\Users\StdUser\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AMD_EquityPrice_2024.01.01.csv'  # Replace with your CSV file path
df_equity_price = pd.read_csv(equity_price_file_path)

# Combine Date and Time into a single datetime column for equity and price data
df_equity_price['DateTime'] = pd.to_datetime(df_equity_price['Date'] + ' ' + df_equity_price['Time'])

# Load transaction data from CSV file
transaction_file_path = r'C:\Users\StdUser\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AMD_2024.01.01_2024.02.28_Mul_indicators_w_DT_SL_Eq_TP_R_9640.csv'  # Replace with your transaction CSV file path
df_transactions = pd.read_csv(transaction_file_path)

# Combine Date and Time into a single datetime column for transaction data
df_transactions['DateTime'] = pd.to_datetime(df_transactions['Date'] + ' ' + df_transactions['Time'])

# Plotting with Plotly
fig = go.Figure()

# Add trace for Bid Price
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
df_buy = df_transactions[df_transactions['Type'].str.contains('Buy')]
df_sell = df_transactions[df_transactions['Type'].str.contains('Sell')]

# Add Buy signals (Green upward triangle)
fig.add_trace(go.Scatter(
    x=df_buy['DateTime'],
    y=df_equity_price.loc[df_equity_price['DateTime'].isin(df_buy['DateTime']), 'Price'],  # Match with equity price DateTime
    mode='markers',
    marker=dict(symbol='triangle-up', color='green', size=10),
    name='Buy',
    hovertemplate=(
        '<b>Buy</b><br>Date: %{x}<br>Price: %{y}<br>'
        'MA Score: %{customdata[0]}<br>MACD Score: %{customdata[1]}<br>RSI Score: %{customdata[2]}<br>'
        'Stoch Score: %{customdata[3]}<br>BB Score: %{customdata[4]}<br>ATR Score: %{customdata[5]}<br>'
        'SAR Score: %{customdata[6]}<br>Ichimoku Score: %{customdata[7]}<br>ADX Score: %{customdata[8]}<br>'
        'Volume Score: %{customdata[9]}<br>Total Score: %{customdata[10]}<extra></extra>'
    ),
    customdata=df_buy[['MA Score', 'MACD Score', 'RSI Score', 'Stoch Score', 'BB Score', 'ATR Score', 'SAR Score', 
                       'Ichimoku Score', 'ADX Score', 'Volume Score', 'Total Score']].values
))

# Add Sell signals (Red inverted triangle)
fig.add_trace(go.Scatter(
    x=df_sell['DateTime'],
    y=df_equity_price.loc[df_equity_price['DateTime'].isin(df_sell['DateTime']), 'Price'],  # Match with equity price DateTime
    mode='markers',
    marker=dict(symbol='triangle-down', color='red', size=10),
    name='Sell',
    hovertemplate=(
        '<b>Sell</b><br>Date: %{x}<br>Price: %{y}<br>'
        'MA Score: %{customdata[0]}<br>MACD Score: %{customdata[1]}<br>RSI Score: %{customdata[2]}<br>'
        'Stoch Score: %{customdata[3]}<br>BB Score: %{customdata[4]}<br>ATR Score: %{customdata[5]}<br>'
        'SAR Score: %{customdata[6]}<br>Ichimoku Score: %{customdata[7]}<br>ADX Score: %{customdata[8]}<br>'
        'Volume Score: %{customdata[9]}<br>Total Score: %{customdata[10]}<extra></extra>'
    ),
    customdata=df_sell[['MA Score', 'MACD Score', 'RSI Score', 'Stoch Score', 'BB Score', 'ATR Score', 'SAR Score', 
                        'Ichimoku Score', 'ADX Score', 'Volume Score', 'Total Score']].values
))

# Update layout to include a secondary y-axis
fig.update_layout(
    title='Equity and Bid Price with Buy/Sell Signals Over Time',
    xaxis_title='DateTime',
    yaxis_title='Bid Price',
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
