import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import sqlite3
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from datetime import datetime
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
import joblib
import json
from model_repository import ModelRepository

def evaluate_arima_model(data: pd.Series, order: tuple) -> Dict:
    """Evaluate an ARIMA model with given parameters"""
    try:
        model = ARIMA(data, order=order)
        results = model.fit()
        predictions = results.fittedvalues
        
        metrics = {
            'aic': results.aic,
            'bic': results.bic,
            'mae': mean_absolute_error(data[1:], predictions[1:]),
            'rmse': np.sqrt(mean_squared_error(data[1:], predictions[1:])),
            'r2': r2_score(data[1:], predictions[1:])
        }
        return order, results, metrics
    except:
        return order, None, None

def auto_arima(data: pd.Series, 
               max_p: int = 5, 
               max_d: int = 2, 
               max_q: int = 5,
               seasonal: bool = True,
               m: int = 5,
               progress_callback=None) -> Tuple[object, Dict]:
    """
    Implement auto ARIMA using statsmodels with grid search
    
    Args:
        data: Time series data
        max_p: Maximum AR order
        max_d: Maximum difference order
        max_q: Maximum MA order
        seasonal: Whether to include seasonal components
        m: Seasonal period
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Tuple of (best model, metrics)
    """
    best_score = float('inf')
    best_model = None
    best_metrics = None
    best_order = None
    best_predictions = None
    
    p = range(0, max_p + 1)
    d = range(0, max_d + 1)
    q = range(0, max_q + 1)
    
    parameters = list(itertools.product(p, d, q))
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for order in parameters:
            futures.append(executor.submit(evaluate_arima_model, data, order))
        
        total = len(parameters)
        completed = 0
        
        for future in as_completed(futures):
            completed += 1
            if progress_callback:
                progress_callback(completed / total, f"Evaluating ARIMA models: {completed}/{total}")
            
            order, results, metrics = future.result()
            if results is not None and metrics is not None:
                score = metrics['aic']
                if score < best_score:
                    best_score = score
                    best_model = results
                    best_metrics = metrics
                    best_order = order
                    best_predictions = results.fittedvalues
    
    if best_model is None:
        raise ValueError("Could not find a suitable ARIMA model")
    
    logging.info(f"Best ARIMA order found: {best_order}")
    best_model.predictions = best_predictions
    return best_model, best_metrics

def train_arima(data: pd.Series, order: tuple) -> Dict:
    """Train ARIMA model"""
    model = ARIMA(data, order=order)
    results = model.fit()
    predictions = results.fittedvalues
    
    metrics = {
        'aic': results.aic,
        'bic': results.bic,
        'mae': mean_absolute_error(data[1:], predictions[1:]),
        'rmse': np.sqrt(mean_squared_error(data[1:], predictions[1:])),
        'r2': r2_score(data[1:], predictions[1:])
    }
    
    return results, metrics

def train_sarima(data: pd.Series, order: tuple, seasonal_order: tuple) -> Dict:
    """Train SARIMA model"""
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    results = model.fit()
    predictions = results.fittedvalues
    
    metrics = {
        'aic': results.aic,
        'bic': results.bic,
        'mae': mean_absolute_error(data[1:], predictions[1:]),
        'rmse': np.sqrt(mean_squared_error(data[1:], predictions[1:])),
        'r2': r2_score(data[1:], predictions[1:])
    }
    
    return results, metrics

def train_prophet(data: pd.DataFrame, features: List[str] = None) -> Dict:
    """Train Prophet model with optional additional regressors"""
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    
    if features:
        for feature in features:
            model.add_regressor(feature)
    
    model.fit(data)
    future = model.make_future_dataframe(periods=0)
    if features:
        for feature in features:
            future[feature] = data[feature].values
    
    predictions = model.predict(future)
    
    metrics = {
        'mae': mean_absolute_error(data['y'], predictions['yhat']),
        'rmse': np.sqrt(mean_squared_error(data['y'], predictions['yhat'])),
        'r2': r2_score(data['y'], predictions['yhat'])
    }
    
    return model, metrics

def check_stationarity(series):
    """Check if a time series is stationary using ADF test"""
    result = adfuller(series.dropna())
    return result[1] < 0.05

def make_stationary(series):
    """Make a time series stationary through differencing"""
    diff_series = series.copy()
    n_diff = 0
    
    while not check_stationarity(diff_series) and n_diff < 2:
        diff_series = diff_series.diff().dropna()
        n_diff += 1
    
    return diff_series, n_diff

def check_constant_series(series):
    """Check if a series is constant (has no variation)"""
    return series.nunique() <= 1

def train_var(data: pd.DataFrame, maxlags: int = 5) -> Dict:
    """Train VAR model with improved preprocessing"""
    logging.info("Starting VAR model training with preprocessing...")
    
    numeric_data = data.select_dtypes(include=[np.number]).copy()
    
    if len(numeric_data.columns) < 2:
        raise ValueError("VAR model requires at least 2 numeric variables")
    
    constant_cols = [col for col in numeric_data.columns if check_constant_series(numeric_data[col])]
    if constant_cols:
        numeric_data = numeric_data.drop(columns=constant_cols)
    
    if len(numeric_data.columns) < 2:
        raise ValueError("Not enough non-constant variables for VAR model")
    
    stationary_data = pd.DataFrame()
    diff_orders = {}
    
    for column in numeric_data.columns:
        series = numeric_data[column]
        series = pd.to_numeric(series, errors='coerce').dropna()
        
        if series.std() < 1e-8:
            series = series + np.random.normal(0, 1e-8, len(series))
        
        try:
            stationary_series, n_diff = make_stationary(series)
            stationary_data[column] = stationary_series
            diff_orders[column] = n_diff
        except Exception as e:
            noisy_series = series + np.random.normal(0, series.std() * 0.01, len(series))
            stationary_series, n_diff = make_stationary(noisy_series)
            stationary_data[column] = stationary_series
            diff_orders[column] = n_diff
    
    stationary_data = stationary_data.dropna()
    
    if len(stationary_data) < 10:
        raise ValueError("Not enough data points after making series stationary")
    
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(stationary_data),
        index=stationary_data.index,
        columns=stationary_data.columns
    )
    
    model = VAR(scaled_data)
    try:
        order = model.select_order(maxlags=maxlags)
        best_order = order.aic.argmin() + 1
    except Exception as e:
        logging.warning(f"Error in order selection: {str(e)}. Using default order of 1")
        best_order = 1
    
    results = model.fit(best_order)
    predictions = pd.DataFrame(
        scaler.inverse_transform(results.fittedvalues),
        index=results.fittedvalues.index,
        columns=results.fittedvalues.columns
    )
    
    for column in predictions.columns:
        for _ in range(diff_orders[column]):
            predictions[column] = predictions[column].cumsum()
            predictions[column] = numeric_data[column].iloc[0] + predictions[column]
    
    metrics = {}
    for col in numeric_data.columns:
        valid_idx = predictions.index.intersection(numeric_data.index)
        metrics[f'{col}_mae'] = mean_absolute_error(
            numeric_data[col].loc[valid_idx], 
            predictions[col].loc[valid_idx]
        )
        metrics[f'{col}_rmse'] = np.sqrt(mean_squared_error(
            numeric_data[col].loc[valid_idx], 
            predictions[col].loc[valid_idx]
        ))
        metrics[f'{col}_r2'] = r2_score(
            numeric_data[col].loc[valid_idx], 
            predictions[col].loc[valid_idx]
        )
    
    metrics.update({
        'order': best_order,
        'diff_orders': diff_orders,
        'n_observations': len(numeric_data)
    })
    
    results.scaler = scaler
    results.diff_orders = diff_orders
    results.predictions = predictions
    
    return results, metrics

def prepare_time_series_data(df: pd.DataFrame, target_col: str, selected_features: List[str], prediction_horizon: int = 1, n_lags: int = 3):
    """Prepare time series data for training"""
    data = df.copy()
    
    for i in range(1, n_lags + 1):
        lag_col = f"{target_col}_lag_{i}"
        data[lag_col] = data[target_col].shift(i)
        if selected_features is not None:
            selected_features.append(lag_col)
    
    future_target = data[target_col].shift(-prediction_horizon)
    data = data[n_lags:-prediction_horizon]
    future_target = future_target[n_lags:-prediction_horizon]
    
    features = data[selected_features] if selected_features else None
    
    return data, future_target, features

def save_model(model, model_path: str, metadata: Dict) -> Tuple[str, Optional[str]]:
    """Save model and its metadata
    
    Args:
        model: The trained model to save
        model_path: Path where to save the model
        metadata: Dictionary containing model metadata
        
    Returns:
        Tuple of (model_path, scaler_path)
    """
    os.makedirs(model_path, exist_ok=True)
    
    # Save model
    model_file_path = os.path.join(model_path, 'model.pkl')
    joblib.dump(model, model_file_path)
    
    # Save scaler if exists
    scaler_path = None
    if hasattr(model, 'scaler'):
        scaler_path = os.path.join(model_path, 'scaler.pkl')
        joblib.dump(model.scaler, scaler_path)
    elif 'scaler' in metadata:
        scaler_path = os.path.join(model_path, 'scaler.pkl')
        joblib.dump(metadata['scaler'], scaler_path)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        return obj
    
    cleaned_metadata = convert_to_native(metadata)
    
    # Save metadata to file
    with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
        json.dump(cleaned_metadata, f, indent=4)
    
    return model_file_path, scaler_path

def combine_tables_data(db_path: str, table_names: List[str]) -> pd.DataFrame:
    """Combine data from multiple tables into a single DataFrame"""
    combined_data = []
    
    try:
        conn = sqlite3.connect(db_path)
        
        for table_name in table_names:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            df['DataSource'] = table_name
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            combined_data.append(df)
            logging.info(f"Loaded {len(df)} rows from {table_name}")
        
        conn.close()
        
        if not combined_data:
            raise ValueError("No data loaded from tables")
            
        combined_df = pd.concat(combined_data, axis=0, ignore_index=True)
        combined_df = combined_df.sort_values('DateTime')
        
        logging.info(f"Combined data shape: {combined_df.shape}")
        return combined_df
        
    except Exception as e:
        logging.error(f"Error combining table data: {str(e)}")
        raise 