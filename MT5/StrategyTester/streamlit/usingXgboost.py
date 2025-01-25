import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from datetime import datetime
import os
import joblib
import json
from model_pipeline import ModelPipeline, create_pipeline_from_analyzer

class EnhancedTimeSeriesAnalyzer:
    def __init__(self, n_splits=5, sequence_length=10):
        self.n_splits = n_splits
        self.sequence_length = sequence_length
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
        self.models = {
            'xgboost': XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'lightgbm': LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        }
        self.results = {}

    def create_time_features(self, df):
        """Create time-based features"""
        df['hour'] = df['DateTime'].dt.hour
        df['day_of_week'] = df['DateTime'].dt.dayofweek
        df['day_of_month'] = df['DateTime'].dt.day
        df['month'] = df['DateTime'].dt.month
        return df

    def create_technical_features(self, df, price_col='Price'):
        """Create technical indicators"""
        # Moving averages
        for window in [5, 10, 20]:
            df[f'ma_{window}'] = df[price_col].rolling(window=window).mean()
            df[f'ma_std_{window}'] = df[price_col].rolling(window=window).std()
        
        # Price momentum
        for window in [5, 10, 20]:
            df[f'momentum_{window}'] = df[price_col].diff(window)
        
        # Relative price features
        df['price_rel_ma5'] = df[price_col] / df['ma_5']
        df['price_rel_ma10'] = df[price_col] / df['ma_10']
        
        return df

    def create_lag_features(self, df, columns, lags=[1, 2, 3]):
        """Create lagged features efficiently"""
        # Create a dictionary to store all new columns
        new_columns = {}
        
        for col in columns:
            for lag in lags:
                new_columns[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Concatenate all new columns at once
        lagged_df = pd.concat([df] + [pd.DataFrame(new_columns)], axis=1)
        return lagged_df

    def prepare_data(self, df, feature_columns, target_column='Price'):
        """Enhanced data preparation with better handling of missing values"""
        # Create a copy of the dataframe to avoid fragmentation
        df = df.copy()
        
        # Convert date and time to datetime
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.sort_values('DateTime')
        
        # Create time features first
        df = self.create_time_features(df)
        
        # Create technical features
        df = self.create_technical_features(df)
        
        # Forward fill any NaN values created by technical indicators
        df = df.ffill()
        
        # Create lag features
        df = self.create_lag_features(df, feature_columns)
        
        # Forward fill any remaining NaN values
        df = df.ffill().bfill()
        
        # Verify we have data after preprocessing
        if len(df) == 0:
            raise ValueError("No data remaining after preprocessing!")
        
        # Prepare features
        feature_cols = (
            feature_columns + 
            [col for col in df.columns if 'ma_' in col or 'momentum_' in col or 'lag' in col] +
            ['hour', 'day_of_week', 'day_of_month', 'month']
        )
        
        # Remove any duplicate columns
        feature_cols = list(dict.fromkeys(feature_cols))
        
        # Ensure all feature columns exist
        existing_cols = [col for col in feature_cols if col in df.columns]
        if len(existing_cols) < len(feature_cols):
            print(f"Warning: Some feature columns were not found in the dataframe. Using {len(existing_cols)} features.")
            feature_cols = existing_cols
        
        X = df[feature_cols].copy()
        y = df[target_column].copy()
        
        # Print shape before scaling
        print(f"Data shape before scaling - X: {X.shape}, y: {y.shape}")
        
        # Scale the data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1))
        
        return X_scaled, y_scaled, df['DateTime'], feature_cols

    def create_lstm_model(self, input_shape):
        """Create enhanced LSTM model"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(100, input_shape=input_shape, return_sequences=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                     loss='huber')  # Huber loss is more robust to outliers
        return model

    def train_and_evaluate(self, X, y, dates):
        """Train and evaluate with walk-forward optimization"""
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=len(X)//10)
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            model_results = []
            predictions = []
            actual_values = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train model
                if model_name == 'xgboost':
                    eval_set = [(X_test, y_test.ravel())]
                    model.set_params(callbacks=[xgb.callback.EarlyStopping(rounds=20)])
                    model.fit(X_train, y_train.ravel(),
                            eval_set=eval_set,
                            verbose=False)
                else:
                    model.fit(X_train, y_train.ravel())
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Store predictions and actual values
                predictions.extend(y_pred)
                actual_values.extend(y_test)
                
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
            
            # Add prediction results
            avg_results['predictions'] = predictions
            avg_results['actual_values'] = actual_values
            
            self.results[model_name] = avg_results

        return self.results

    def evaluate_model(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate directional accuracy
        direction_correct = np.mean(
            (np.diff(y_true.ravel()) * np.diff(y_pred.ravel())) > 0
        )
        
        return {
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'MAE': mae,
            'R2': r2,
            'DirectionalAccuracy': direction_correct
        }

def plot_detailed_results(analyzer_results, dates):
    """Plot detailed comparison of model results with numerical summaries"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # First, display numerical summaries
    metrics = ['MSE', 'RMSE', 'MAE', 'R2', 'DirectionalAccuracy']
    
    # Create a DataFrame for easy comparison
    results_df = pd.DataFrame()
    for model_name, results in analyzer_results.items():
        model_metrics = {metric: results[metric] for metric in metrics}
        results_df[model_name] = pd.Series(model_metrics)
    
    # Display numerical results
    print("\nNumerical Results Summary:")
    print("=" * 80)
    print(results_df.round(4))
    print("\nBest Model per Metric:")
    print("-" * 80)
    for metric in metrics:
        if metric == 'MSE' or metric == 'RMSE' or metric == 'MAE':
            best_model = results_df.loc[metric].idxmin()
            best_value = results_df.loc[metric].min()
        else:
            best_model = results_df.loc[metric].idxmax()
            best_value = results_df.loc[metric].max()
        print(f"{metric:20} : {best_model:15} (Value: {best_value:.4f})")
    print("=" * 80)
    
    # Plot metrics comparison
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        values = [results[metric] for results in analyzer_results.values()]
        ax = sns.barplot(x=list(analyzer_results.keys()), y=values)
        plt.title(f'Comparison of {metric} across models')
        plt.xticks(rotation=45)
        
        # Add value labels on top of each bar
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    # Plot actual vs predicted for each model
    for model_name, results in analyzer_results.items():
        plt.figure(figsize=(15, 6))
        
        # Get actual and predicted values
        actuals = np.array(results['actual_values'])
        preds = np.array(results['predictions'])
        
        # Plot time series
        plt.subplot(1, 2, 1)
        plt.plot(dates[-len(actuals):], actuals, label='Actual', alpha=0.7)
        plt.plot(dates[-len(preds):], preds, label='Predicted', alpha=0.7)
        plt.title(f'{model_name} - Time Series Comparison')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Plot scatter plot
        plt.subplot(1, 2, 2)
        plt.scatter(actuals, preds, alpha=0.5)
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', alpha=0.7)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name} - Actual vs Predicted Scatter')
        
        plt.tight_layout()
        plt.show()
        
        # Print correlation coefficient
        correlation = np.corrcoef(actuals.ravel(), preds.ravel())[0, 1]
        print(f"\n{model_name} - Correlation coefficient: {correlation:.4f}")
        
        # Print additional statistics
        mean_actual = np.mean(actuals)
        mean_pred = np.mean(preds)
        std_actual = np.std(actuals)
        std_pred = np.std(preds)
        
        print(f"Statistics for {model_name}:")
        print(f"{'':4}Mean - Actual: {mean_actual:.4f}, Predicted: {mean_pred:.4f}")
        print(f"{'':4}Std  - Actual: {std_actual:.4f}, Predicted: {std_pred:.4f}")
        print("-" * 50)


# class ModelPipeline:
#     def __init__(self):
#         self.model = None
#         self.feature_scaler = None
#         self.target_scaler = None
#         self.feature_columns = None
        
#     def save_pipeline(self, model, feature_scaler, target_scaler, feature_columns, base_path='models'):
#         """Save all components needed for prediction"""
#         # Create dictionary of components
#         pipeline_components = {
#             'model': model,
#             'feature_scaler': feature_scaler,
#             'target_scaler': target_scaler
#         }
        
#         # Save scalers and model
#         joblib.dump(pipeline_components, f'{base_path}/model_pipeline.joblib')
        
#         # Save feature columns
#         with open(f'{base_path}/feature_columns.json', 'w') as f:
#             json.dump(feature_columns, f)
            
#         print(f"Model pipeline saved to {base_path}/")



# def save_best_model(analyzer, X, feature_columns, base_path='models'):
#     """Save the best model (XGBoost in this case)"""
#     pipeline = ModelPipeline()
#     pipeline.save_pipeline(
#         model=analyzer.models['xgboost'],
#         feature_scaler=analyzer.scaler_X,
#         target_scaler=analyzer.scaler_y,
#         feature_columns=feature_columns,
#         base_path=base_path
#     )
#     return pipeline



def save_best_model(analyzer, X, feature_columns, base_path='models'):
    """Save the best model using the enhanced pipeline"""
    return create_pipeline_from_analyzer(
        analyzer=analyzer,
        X=X,
        feature_columns=feature_columns,
        base_path=base_path
    )


# Example usage
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
    analyzer = EnhancedTimeSeriesAnalyzer(n_splits=5)
    
    # Prepare data
    X, y, dates, feature_cols = analyzer.prepare_data(df, feature_columns)
    
    # Train and evaluate models
    results = analyzer.train_and_evaluate(X, y, dates)
    
    # Plot detailed results
    plot_detailed_results(results, dates)

    # Create models directory if it doesn't exist
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Save the model pipeline
    pipeline = save_best_model(analyzer, X, feature_cols, base_path=model_dir)
    print(f"Model saved successfully in {model_dir}")