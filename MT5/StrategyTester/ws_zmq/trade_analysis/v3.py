# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
def prepare_data(file_path):
    # Load data
    parsed_df = pd.read_csv(file_path)
    
    # Define columns to exclude
    columns_to_exclude = ['Timestamp', 'OrderType', 'Symbol', 'OrderID']
    parsed_df_filtered = parsed_df[[col for col in parsed_df.columns if col not in columns_to_exclude]]
    
    # Prepare features and target
    X = parsed_df_filtered.drop(columns=['CurrentProfit'])
    y = parsed_df_filtered['CurrentProfit']
    
    return X, y

def train_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    
    return rf, X_train, X_test, y_train, y_test, y_pred

def create_visualizations(rf, X, y_test, y_pred, feature_importance):
    plt.style.use('seaborn')
    
    # Create subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Feature Importance Bar Plot
    sns.barplot(
        data=feature_importance.head(10),
        x='importance',
        y='feature',
        ax=axes[0,0]
    )
    axes[0,0].set_title('Top 10 Feature Importance')
    axes[0,0].set_xlabel('Importance Score')
    
    # 2. Actual vs Predicted Plot
    axes[0,1].scatter(y_test, y_pred, alpha=0.5)
    axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0,1].set_xlabel('Actual Values')
    axes[0,1].set_ylabel('Predicted Values')
    axes[0,1].set_title('Actual vs Predicted')
    
    # 3. Residual Plot
    residuals = y_test - y_pred
    axes[1,0].scatter(y_pred, residuals, alpha=0.5)
    axes[1,0].axhline(y=0, color='r', linestyle='--')
    axes[1,0].set_xlabel('Predicted Values')
    axes[1,0].set_ylabel('Residuals')
    axes[1,0].set_title('Residual Plot')
    
    # 4. Correlation Heatmap
    top_features = feature_importance['feature'].head(10).tolist()
    correlation_matrix = X[top_features].corr()
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        cmap='coolwarm', 
        ax=axes[1,1],
        fmt='.2f'
    )
    axes[1,1].set_title('Feature Correlation Heatmap')
    
    plt.tight_layout()
    plt.savefig('trading_analysis_visualization.png')
    
    # Feature Importance Distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=feature_importance['importance'], fill=True)
    plt.title('Feature Importance Distribution')
    plt.xlabel('Importance Score')
    plt.ylabel('Density')
    plt.savefig('importance_distribution.png')
    
    plt.show()

def main():
    # Load and prepare data
    file_path = r"C:\Users\StdUser\Desktop\MyProjects\Backtesting\logs\SYM_10027915_transactions.csv"  # Replace with your data file path
    X, y = prepare_data(file_path)
    
    # Train model and get predictions
    rf, X_train, X_test, y_train, y_test, y_pred = train_model(X, y)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    # Print metrics
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save feature importance
    feature_importance.to_csv('feature_importances.csv', index=False)
    
    # Create visualizations
    create_visualizations(rf, X, y_test, y_pred, feature_importance)

if __name__ == "__main__":
    main()