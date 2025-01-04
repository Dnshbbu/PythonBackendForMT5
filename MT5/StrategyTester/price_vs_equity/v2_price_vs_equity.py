import pandas as pd
import plotly.graph_objects as go

# Load data from CSV file
file_path = r'C:\Users\StdUser\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AMD_EquityPrice_2024.04.01.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Combine Date and Time into a single datetime column
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

# Plotting with Plotly
fig = go.Figure()

# Add trace for Bid Price with markers
fig.add_trace(go.Scatter(
    x=df['DateTime'],
    y=df['Price'],
    # mode='lines+markers',
    mode='lines',
    name='Price',
    yaxis='y1',
    hovertemplate='<b>Bid Price</b><br>Date: %{x}<br>Price: %{y}<extra></extra>'  # Custom hover template
))

# Add trace for Equity with markers
fig.add_trace(go.Scatter(
    x=df['DateTime'],
    y=df['Equity'],
    # mode='lines+markers',
    mode='lines',
    name='Equity',
    yaxis='y2',
    hovertemplate='<b>Equity</b><br>Date: %{x}<br>Equity: %{y}<extra></extra>'  # Custom hover template
))

# Update layout to include a secondary y-axis
fig.update_layout(
    title='Equity and Bid Price Over Time',
    xaxis_title='DateTime',
    yaxis_title='Bid Price',
    yaxis2=dict(title='Equity', overlaying='y', side='right'),
    xaxis_rangeslider_visible=True,
    hovermode='x unified',  # Enable unified hover mode
)

# Set x-axis range based on your data
min_date = df['DateTime'].min()
max_date = df['DateTime'].max()
fig.update_xaxes(range=[min_date, max_date])  # Set the x-axis range

# Save the figure to an HTML file
output_file_path = 'output_plot.html'  # Specify the output HTML file name
fig.write_html(output_file_path)

print(f"Plot saved as {output_file_path}")
