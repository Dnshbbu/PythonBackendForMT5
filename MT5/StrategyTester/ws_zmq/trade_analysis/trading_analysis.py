import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Debug step 1: Load and verify data
df = pd.read_csv(r"C:\Users\StdUser\Desktop\MyProjects\Backtesting\logs\SYM_10016570_transactions.csv")
print(f"Initial DataFrame shape: {df.shape}")

# Parse trading factors
def parse_factors(x):
    if pd.isna(x):
        return {}
    result = {}
    pairs = [p.split('=') for p in str(x).split('|') if '=' in p]
    for k, v in pairs:
        try:
            result[k] = float(v)
        except:
            result[k] = 0
    return result

# Debug step 2: Process features
print("\nProcessing features...")
factors_dict = df['factors'].apply(parse_factors).to_dict()
score_dict = df['score'].apply(parse_factors).to_dict()
exit_dict = df['exitScore'].apply(parse_factors).to_dict()

# Convert to DataFrames
factors_df = pd.DataFrame.from_dict(factors_dict, orient='index')
score_df = pd.DataFrame.from_dict(score_dict, orient='index')
exit_df = pd.DataFrame.from_dict(exit_dict, orient='index')

print(f"Factors shape: {factors_df.shape}")
print(f"Score shape: {score_df.shape}")
print(f"Exit shape: {exit_df.shape}")

# Combine features
df = pd.concat([df, factors_df, score_df, exit_df], axis=1)

# Prepare features for model
exclude_cols = ['Time', 'Ticket', 'Action', 'Symbol', 'CloseReason', 
                'factors', 'score', 'efactors', 'exitScore']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].fillna(0)
y = df['CurrentProfit']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Output results
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Save results
feature_importance.to_csv('feature_importances.csv', index=False)
print("\nResults saved to feature_importances.csv")