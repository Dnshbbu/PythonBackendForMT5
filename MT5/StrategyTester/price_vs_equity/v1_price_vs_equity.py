import pandas as pd
import plotly.graph_objects as go

# Load data from CSV file
file_path = r'C:\Users\StdUser\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EquityPrice_BABA_2024.01.01.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Combine Date and Time into a single datetime column
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

# Plotting with Plotly
fig = go.Figure()

# Add trace for Bid Price with markers
fig.add_trace(go.Scatter(
    x=df['DateTime'],
    y=df['Bid_Price'],
    mode='lines+markers',
    name='Bid Price',
    yaxis='y1',
    hovertemplate='<b>Bid Price</b><br>Date: %{x}<br>Price: %{y}<extra></extra>'  # Custom hover template
))

# Add trace for Equity with markers
fig.add_trace(go.Scatter(
    x=df['DateTime'],
    y=df['Equity'],
    mode='lines+markers',
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
    shapes=[
        # Vertical line shape (crosshair)
        dict(
            type='line',
            x0=0,  # Placeholder, will update in Dash
            y0=0,
            x1=0,  # Placeholder, will update in Dash
            y1=1,
            xref='x',
            yref='paper',
            line=dict(color='black', width=1, dash='dash'),  # Dotted vertical line
            layer='above'
        ),
        # Horizontal line shape (crosshair)
        dict(
            type='line',
            x0=0,  # Placeholder, will update in Dash
            y0=0,  # Placeholder, will update in Dash
            x1=1,
            y1=0,  # Placeholder, will update in Dash
            xref='paper',
            yref='y',
            line=dict(color='black', width=1, dash='dash'),  # Dotted horizontal line
            layer='above'
        )
    ]
)

# Save the figure to an HTML file
output_file_path = 'output_plot.html'  # Specify the output HTML file name
fig.write_html(output_file_path)

print(f"Plot saved as {output_file_path}")
