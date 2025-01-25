import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesAnalyzer:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': LGBMRegressor(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.best_score = float('-inf')
        
    def prepare_data(self, df, feature_columns, target_column):
        """Prepare data for time series analysis"""
        # Create feature matrix and target vector
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill')
        y = y.fillna(method='ffill').fillna(method='bfill')
        
        return X, y
    
    def create_sequences(self, X, y, lookback=5):
        """Create sequences for time series prediction"""
        X_seq, y_seq = [], []
        for i in range(len(X) - lookback):
            X_seq.append(X.iloc[i:i+lookback].values)
            y_seq.append(y.iloc[i+lookback])
        return np.array(X_seq), np.array(y_seq)
    
    def evaluate_models(self, X, y):
        """Evaluate multiple models using time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        results = {}
        
        for name, model in self.models.items():
            mse_scores = []
            r2_scores = []
            
            for train_idx, test_idx in tscv.split(X):
                # Split data
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Scale data
                X_train_scaled = self.scaler_X.fit_transform(X_train)
                X_test_scaled = self.scaler_X.transform(X_test)
                y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
                
                # Train and predict
                model.fit(X_train_scaled, y_train_scaled)
                y_pred_scaled = model.predict(X_test_scaled)
                y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                mse_scores.append(mse)
                r2_scores.append(r2)
            
            # Store results
            results[name] = {
                'MSE': np.mean(mse_scores),
                'R2': np.mean(r2_scores),
                'MSE_std': np.std(mse_scores),
                'R2_std': np.std(r2_scores)
            }
            
            # Update best model
            if np.mean(r2_scores) > self.best_score:
                self.best_score = np.mean(r2_scores)
                self.best_model = model
        
        return results
    
    def train_best_model(self, X, y):
        """Train the best model on the full dataset"""
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        self.best_model.fit(X_scaled, y_scaled)
    
    def predict(self, X_new):
        """Make predictions using the best model"""
        X_scaled = self.scaler_X.transform(X_new)
        y_pred_scaled = self.best_model.predict(X_scaled)
        return self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# Usage example
def run_analysis(csv_path):
    # Read data
    df = pd.read_csv(csv_path)
    
    # Define features and target
    feature_columns = ['EntryScore_SR', 'EntryScore_Pullback', 'EntryScore_EMA', 'EntryScore_AVWAP']
    target_column = 'Price'
    
    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer(n_splits=5)
    
    # Prepare data
    X, y = analyzer.prepare_data(df, feature_columns, target_column)
    
    # Create sequences for time series analysis
    X_seq, y_seq = analyzer.create_sequences(X, y, lookback=5)
    
    # Evaluate models
    results = analyzer.evaluate_models(X_seq.reshape(X_seq.shape[0], -1), y_seq)
    
    # Print results
    print("\nModel Evaluation Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"MSE: {metrics['MSE']:.4f} (±{metrics['MSE_std']:.4f})")
        print(f"R2: {metrics['R2']:.4f} (±{metrics['R2_std']:.4f})")
    
    # Train best model on full dataset
    analyzer.train_best_model(X_seq.reshape(X_seq.shape[0], -1), y_seq)
    
    return analyzer

if __name__ == "__main__":
    csv_path = r"C:\Users\StdUser\Desktop\MyProjects\Backtesting\MT5\StrategyTester\streamlit\processed_data\SYM_10028030_all_details_processed_20250124_234257.csv"
    analyzer = run_analysis(csv_path)