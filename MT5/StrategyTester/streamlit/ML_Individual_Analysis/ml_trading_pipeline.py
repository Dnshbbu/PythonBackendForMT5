import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

class PricePredictionPipeline:
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
    def prepare_data(self, df):
        feature_columns = [
            'EntryScore_SR',
            'EntryScore_Pullback',
            'EntryScore_EMA',
            'EntryScore_AVWAP'
        ]
        
        X = df[feature_columns]
        y = df['Price']
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        return X, y
    
    def train_evaluate_models(self, X, y, test_size=0.2):
        # Impute missing values
        X_imputed = self.imputer.fit_transform(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        train_index, test_index = list(tscv.split(X_scaled))[-1]
        
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        results = {}
        
        for name, model in self.models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                results[name] = {
                    'r2': r2_score(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'model': model,
                    'feature_importance': self._get_feature_importance(model, X.columns)
                }
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
            
        return results, X_test, y_test
    
    def _get_feature_importance(self, model, feature_names):
        if hasattr(model, 'coef_'):
            importance = model.coef_
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            return None
        
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)

def run_price_prediction(csv_path):
    # Load data
    df = pd.read_csv(csv_path)
    
    # Initialize pipeline
    pipeline = PricePredictionPipeline()
    
    # Prepare data
    X, y = pipeline.prepare_data(df)
    
    print(f"Features shape: {X.shape}")
    print(f"Missing values:\n{X.isnull().sum()}")
    
    # Train and evaluate models
    results, X_test, y_test = pipeline.train_evaluate_models(X, y)
    
    # Print results
    for name, result in results.items():
        print(f"\n{name.upper()} Results:")
        print(f"R² Score: {result['r2']:.3f}")
        print(f"MSE: {result['mse']:.3f}")
        
        if result['feature_importance'] is not None:
            print("\nFeature Importance:")
            print(result['feature_importance'])
    
    return pipeline, results, X_test, y_test

if __name__ == "__main__":
    pipeline, results, X_test, y_test = run_price_prediction(r"C:\Users\StdUser\Desktop\MyProjects\Backtesting\MT5\StrategyTester\streamlit\processed_data\SYM_10028030_all_details_processed_20250124_234257.csv")