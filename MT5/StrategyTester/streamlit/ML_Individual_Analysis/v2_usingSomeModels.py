import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from datetime import datetime

class TimeSeriesAnalyzer:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.models = {
            'xgboost': XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'lightgbm': LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        }
        self.results = {}

    def prepare_data(self, df, feature_columns, target_column='Price'):
        """Prepare data for modeling"""
        # Convert date and time to datetime
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.sort_values('DateTime')

        # Prepare features and target
        X = df[feature_columns].copy()
        y = df[target_column].copy()

        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        # Scale the data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1))

        return X_scaled, y_scaled, df['DateTime']

    def create_sequences(self, X, y, sequence_length=10):
        """Create sequences for time series prediction"""
        X_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i+sequence_length])
            y_seq.append(y[i+sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def create_lstm_model(self, input_shape):
        """Create LSTM model"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, input_shape=input_shape, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def evaluate_model(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'MAE': mae,
            'R2': r2
        }

    def train_and_evaluate(self, X, y, dates):
        """Train and evaluate multiple models"""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            model_results = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train model
                model.fit(X_train, y_train.ravel())
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Inverse transform predictions
                y_test_orig = self.scaler_y.inverse_transform(y_test)
                y_pred_orig = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1))
                
                # Evaluate
                metrics = self.evaluate_model(y_test_orig, y_pred_orig)
                model_results.append(metrics)
            
            # Average results across folds
            avg_results = {
                metric: np.mean([fold[metric] for fold in model_results])
                for metric in model_results[0].keys()
            }
            self.results[model_name] = avg_results

        return self.results

    def train_lstm(self, X, y, sequence_length=10, epochs=50, batch_size=32):
        """Train and evaluate LSTM model"""
        X_seq, y_seq = self.create_sequences(X, y, sequence_length)
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        lstm_results = []
        
        for train_idx, test_idx in tscv.split(X_seq):
            X_train, X_test = X_seq[train_idx], X_seq[test_idx]
            y_train, y_test = y_seq[train_idx], y_seq[test_idx]
            
            # Create and train LSTM model
            model = self.create_lstm_model((sequence_length, X.shape[1]))
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Inverse transform predictions
            y_test_orig = self.scaler_y.inverse_transform(y_test)
            y_pred_orig = self.scaler_y.inverse_transform(y_pred)
            
            # Evaluate
            metrics = self.evaluate_model(y_test_orig, y_pred_orig)
            lstm_results.append(metrics)
        
        # Average results across folds
        avg_results = {
            metric: np.mean([fold[metric] for fold in lstm_results])
            for metric in lstm_results[0].keys()
        }
        self.results['lstm'] = avg_results
        
        return self.results

def plot_results(analyzer_results):
    """Plot comparison of model results"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(12, 6))
    metrics = list(next(iter(analyzer_results.values())).keys())
    
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        values = [results[metric] for results in analyzer_results.values()]
        sns.barplot(x=list(analyzer_results.keys()), y=values)
        plt.title(f'Comparison of {metric} across models')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Usage example
if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv(r"C:\Users\StdUser\Desktop\MyProjects\Backtesting\MT5\StrategyTester\streamlit\processed_data\SYM_10029174_all_details_processed_20250125_145908.csv")
    
    # Define feature columns
    feature_columns = [
        'EntryScore_SR', 'EntryScore_Pullback', 'EntryScore_EMA', 'EntryScore_AVWAP',
        'Factors_srScore', 'Factors_maScore', 'Factors_rsiScore', 'Factors_macdScore',
        'Factors_stochScore', 'Factors_bbScore', 'Factors_atrScore', 'Factors_sarScore',
        'Factors_ichimokuScore', 'Factors_adxScore', 'Factors_volumeScore', 'Factors_mfiScore',
        'Factors_priceMAScore', 'Factors_emaScore', 'Factors_emaCrossScore', 'Factors_cciScore'
    ]
    
    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer(n_splits=5)
    
    # Prepare data
    X, y, dates = analyzer.prepare_data(df, feature_columns)
    
    # Train and evaluate traditional models
    results = analyzer.train_and_evaluate(X, y, dates)
    
    # Train and evaluate LSTM
    results = analyzer.train_lstm(X, y)
    
    # Plot results
    plot_results(results)