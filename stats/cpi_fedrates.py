import pandas as pd
from fredapi import Fred

# Replace 'your_api_key' with your actual FRED API key
fred = Fred(api_key='c18fd978df29c2a606c588977da9823e')

# Fetching data from FRED API
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

# Calculate the changes in Fed Rate and CPI between periods
merged_data['Fed_Rate_Change'] = merged_data['Fed_Rate'].diff()
merged_data['CPI_Change'] = merged_data['CPI'].pct_change() * 100  # CPI % change

# Classify periods of Fed rate increases and decreases
merged_data['Rate_Direction'] = merged_data['Fed_Rate_Change'].apply(lambda x: 'Increase' if x > 0 else ('Decrease' if x < 0 else 'No Change'))

# Filter out 'No Change' periods
rate_increase_periods = merged_data[merged_data['Rate_Direction'] == 'Increase']
rate_decrease_periods = merged_data[merged_data['Rate_Direction'] == 'Decrease']

# Calculate statistics for CPI during rate increase and decrease periods
increase_cpi_mean = rate_increase_periods['CPI_Change'].mean()
decrease_cpi_mean = rate_decrease_periods['CPI_Change'].mean()

# Count the number of periods
increase_count = len(rate_increase_periods)
decrease_count = len(rate_decrease_periods)

# Output the results
print(f"Average CPI Change during Fed Rate Increases: {increase_cpi_mean:.2f}%")
print(f"Number of Rate Increase Periods: {increase_count}")
print(f"Average CPI Change during Fed Rate Decreases: {decrease_cpi_mean:.2f}%")
print(f"Number of Rate Decrease Periods: {decrease_count}")
