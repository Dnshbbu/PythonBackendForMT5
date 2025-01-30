import pandas as pd
import plotly.graph_objs as go
from fredapi import Fred

# Replace 'your_api_key' with your actual FRED API key
fred = Fred(api_key='c18fd978df29c2a606c588977da9823e')

# Fetching data from FRED API
# CPI: Consumer Price Index for All Urban Consumers (CPIAUCSL)
# Fed Funds Rate: Effective Federal Funds Rate (FEDFUNDS)
cpi_data = fred.get_series('CPIAUCSL')
fed_rate_data = fred.get_series('FEDFUNDS')

# Convert the fetched data to DataFrame and reset index
cpi_df = pd.DataFrame(cpi_data, columns=['CPI'])
cpi_df.reset_index(inplace=True)
cpi_df.rename(columns={'index': 'Date'}, inplace=True)

fed_rate_df = pd.DataFrame(fed_rate_data, columns=['Fed_Rate'])
fed_rate_df.reset_index(inplace=True)
fed_rate_df.rename(columns={'index': 'Date'}, inplace=True)

# Merging data on 'Date' for common timeline
merged_data = pd.merge(cpi_df, fed_rate_df, on='Date', how='inner')

# Normalize the data to the first available value
# This will allow us to compare percentage changes
merged_data['CPI_Normalized'] = (merged_data['CPI'] / merged_data['CPI'].iloc[0]) * 100
merged_data['Fed_Rate_Normalized'] = (merged_data['Fed_Rate'] / merged_data['Fed_Rate'].iloc[0]) * 100

# Create Plotly traces for normalized CPI and Fed Interest Rate
fig = go.Figure()

# CPI trace (normalized)
fig.add_trace(go.Scatter(
    x=merged_data['Date'], 
    y=merged_data['CPI_Normalized'], 
    mode='lines', 
    name='CPI (Normalized)',
    line=dict(color='blue')
))

# Fed Interest Rate trace (normalized)
fig.add_trace(go.Scatter(
    x=merged_data['Date'], 
    y=merged_data['Fed_Rate_Normalized'], 
    mode='lines', 
    name='Fed Interest Rate (Normalized)',
    line=dict(color='red')
))

# Update layout with titles and axis labels
fig.update_layout(
    title='Normalized CPI vs. Federal Interest Rate',
    xaxis_title='Date',
    yaxis_title='Normalized Value (Percentage Change)',
    legend_title='Indicators',
    hovermode='x unified'
)

# Show the plot
fig.show()
