import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
plt.style.use('seaborn')
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

# 4. Correlation Heatmap of Top Features
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
plt.show()

# Additional distribution plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=feature_importance['importance'], fill=True)
plt.title('Feature Importance Distribution')
plt.xlabel('Importance Score')
plt.ylabel('Density')
plt.savefig('importance_distribution.png')
plt.show()