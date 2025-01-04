import pandas as pd
import plotly.graph_objects as go

# Load data from CSV file
file_path = r'C:\Users\StdUser\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EquityPrice_BABA_2024.01.01.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Combine Date and Time into a single datetime column
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

# Plotting with Plotly
fig = go.Figure()

# Add trace for Bid Price
fig.add_trace(go.Scatter(x=df['DateTime'], y=df['Bid_Price'], mode='lines', name='Bid Price'))

# Add trace for Equity
fig.add_trace(go.Scatter(x=df['DateTime'], y=df['Equity'], mode='lines', name='Equity'))

# Update layout
fig.update_layout(title='Equity and Bid Price Over Time',
                  xaxis_title='DateTime',
                  yaxis_title='Value',
                  xaxis_rangeslider_visible=True)

# Save the figure to an HTML file
output_file_path = 'output_plot.html'  # Specify the output HTML file name
fig.write_html(output_file_path)

print(f"Plot saved as {output_file_path}")
