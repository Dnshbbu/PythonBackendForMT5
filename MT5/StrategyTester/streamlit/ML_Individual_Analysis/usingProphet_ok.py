import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class TimeSeriesPrediction:
    def __init__(self):
        self.model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            seasonality_mode='multiplicative'
        )
        
    def prepare_data(self, df):
        # Prophet requires columns named 'ds' (date) and 'y' (target)
        df_prophet = pd.DataFrame()
        df_prophet['ds'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df_prophet['y'] = df['Price']
        
        # Add regressor columns
        regressors = ['EntryScore_SR', 'EntryScore_Pullback', 
                     'EntryScore_EMA', 'EntryScore_AVWAP',
                     'Factors_srScore',	'Factors_maScore',	'Factors_rsiScore',	'Factors_macdScore',	'Factors_stochScore',	'Factors_bbScore',	'Factors_atrScore',	'Factors_sarScore',	'Factors_ichimokuScore',	'Factors_adxScore',	'Factors_volumeScore',	'Factors_mfiScore',	'Factors_priceMAScore',	'Factors_emaScore',	'Factors_emaCrossScore',	'Factors_cciScore',
]
        
        for reg in regressors:
            df_prophet[reg] = df[reg]
            self.model.add_regressor(reg)
            
        return df_prophet
        
    def train_evaluate(self, df, train_size=0.8):
        # Split data
        train_df = df.iloc[:int(len(df) * train_size)]
        test_df = df.iloc[int(len(df) * train_size):]
        
        # Fit model
        self.model.fit(train_df)
        
        # Make predictions
        forecast = self.model.predict(test_df)
        
        # Calculate metrics
        y_true = test_df['y'].values
        y_pred = forecast['yhat'].values
        
        results = {
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'forecast': forecast,
            'test_actual': y_true
        }
        
        return results
    
    def plot_results(self, results):
        plt.figure(figsize=(12, 6))
        
        # Use the datetime index for plotting
        dates = results['forecast']['ds']
        
        plt.plot(dates, results['test_actual'], label='Actual', marker='o')
        plt.plot(dates, results['forecast']['yhat'], label='Predicted', marker='x')
        plt.fill_between(
            dates,
            results['forecast']['yhat_lower'],
            results['forecast']['yhat_upper'],
            alpha=0.3,
            label='Confidence Interval'
        )
        
        plt.legend()
        plt.title('Price Prediction Results')
        plt.xlabel('Date')
        plt.ylabel('Price')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        plt.show()

def run_timeseries_prediction(csv_path):
    # Load data
    df = pd.read_csv(csv_path)
    
    # Initialize model
    predictor = TimeSeriesPrediction()
    
    # Prepare data
    df_prophet = predictor.prepare_data(df)
    
    # Train and evaluate
    results = predictor.train_evaluate(df_prophet)
    
    print(f"R² Score: {results['r2']:.3f}")
    print(f"MSE: {results['mse']:.3f}")
    
    # Plot results
    predictor.plot_results(results)
    
    return predictor, results

if __name__ == "__main__":
    predictor, results = run_timeseries_prediction(r"C:\Users\StdUser\Desktop\MyProjects\Backtesting\MT5\StrategyTester\streamlit\processed_data\SYM_10029174_all_details_processed_20250125_145908.csv")