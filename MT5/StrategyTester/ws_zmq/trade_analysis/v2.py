import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the CSV file
df = pd.read_csv(r"C:\Users\StdUser\Desktop\MyProjects\Backtesting\logs\SYM_10016570_transactions.csv")
print(f"Initial DataFrame shape: {df.shape}")

# Function to parse key-value pairs from columns like 'factors', 'score', etc.
def parse_factors(column):
    parsed_data = df[column].apply(lambda x: {} if pd.isna(x) else {
        k: float(v) if v.replace('.', '', 1).isdigit() else 0
        for k, v in (item.split('=') for item in str(x).split('|') if '=' in item)
    })
    return pd.DataFrame.from_records(parsed_data)

# Parse all relevant columns
factors_df = parse_factors('factors')
score_df = parse_factors('score')
efactors_df = parse_factors('efactors')
exit_score_df = parse_factors('exitScore')

# Combine parsed DataFrames into the main DataFrame
parsed_df = pd.concat([factors_df, score_df, efactors_df, exit_score_df], axis=1).fillna(0)

# Add the 'CurrentProfit' column as the target
parsed_df['CurrentProfit'] = df['CurrentProfit']

# Exclude specific columns from analysis
columns_to_exclude = ['Threshold', 'FinalScore']
parsed_df_filtered = parsed_df[[col for col in parsed_df.columns if col not in columns_to_exclude]]

# Prepare features (X) and target (y)
X = parsed_df_filtered.drop(columns=['CurrentProfit'])
y = parsed_df_filtered['CurrentProfit']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values(by='importance', ascending=False)

# Output top 10 most important features
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Save feature importance to a CSV
feature_importance.to_csv('feature_importances.csv', index=False)
print("\nFeature importance saved to feature_importances.csv")
