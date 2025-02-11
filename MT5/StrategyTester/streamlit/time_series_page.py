import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import os
import sqlite3
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from datetime import datetime
from db_info import get_table_names, get_numeric_columns
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

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
               m: int = 5) -> Tuple[object, Dict]:
    """
    Implement auto ARIMA using statsmodels with grid search
    
    Args:
        data: Time series data
        max_p: Maximum AR order
        max_d: Maximum difference order
        max_q: Maximum MA order
        seasonal: Whether to include seasonal components
        m: Seasonal period
        
    Returns:
        Tuple of (best model, metrics)
    """
    best_score = float('inf')
    best_model = None
    best_metrics = None
    best_order = None
    
    # Create parameter grid
    p = range(0, max_p + 1)
    d = range(0, max_d + 1)
    q = range(0, max_q + 1)
    
    parameters = list(itertools.product(p, d, q))
    
    # Progress bar for grid search
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for order in parameters:
            futures.append(
                executor.submit(evaluate_arima_model, data, order)
            )
        
        total = len(parameters)
        completed = 0
        
        # Collect results
        for future in as_completed(futures):
            completed += 1
            progress = completed / total
            progress_bar.progress(progress)
            status_text.text(f"Evaluating ARIMA models: {completed}/{total}")
            
            order, results, metrics = future.result()
            if results is not None and metrics is not None:
                # Use AIC as the criterion
                score = metrics['aic']
                if score < best_score:
                    best_score = score
                    best_model = results
                    best_metrics = metrics
                    best_order = order
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    if best_model is None:
        raise ValueError("Could not find a suitable ARIMA model")
    
    st.info(f"Best ARIMA order found: {best_order}")
    return best_model, best_metrics

def train_arima(data: pd.Series, order: tuple) -> Dict:
    """Train ARIMA model"""
    model = ARIMA(data, order=order)
    results = model.fit()
    
    # Make in-sample predictions
    predictions = results.fittedvalues
    
    # Calculate metrics
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
    
    # Make in-sample predictions
    predictions = results.fittedvalues
    
    # Calculate metrics
    metrics = {
        'aic': results.aic,
        'bic': results.bic,
        'mae': mean_absolute_error(data[1:], predictions[1:]),
        'rmse': np.sqrt(mean_squared_error(data[1:], predictions[1:])),
        'r2': r2_score(data[1:], predictions[1:])
    }
    
    return results, metrics

def train_prophet(data: pd.DataFrame) -> Dict:
    """Train Prophet model"""
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    model.fit(data)
    
    # Make in-sample predictions
    future = model.make_future_dataframe(periods=0)
    predictions = model.predict(future)
    
    # Calculate metrics
    metrics = {
        'mae': mean_absolute_error(data['y'], predictions['yhat']),
        'rmse': np.sqrt(mean_squared_error(data['y'], predictions['yhat'])),
        'r2': r2_score(data['y'], predictions['yhat'])
    }
    
    return model, metrics

def time_series_page():
    """Streamlit page for time series models"""
    st.markdown("""
        <h2 style='text-align: center;'>📈 Time Series Models</h2>
        <p style='text-align: center; color: #666;'>
            Train specialized time series models for prediction
        </p>
        <hr>
    """, unsafe_allow_html=True)
    
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
    models_dir = os.path.join(current_dir, 'models', 'time_series')
    os.makedirs(models_dir, exist_ok=True)
    
    # Create main left-right layout
    left_col, right_col = st.columns([1, 1], gap="large")
    
    # Left Column - Configuration
    with left_col:
        st.markdown("#### 📊 Data Configuration")
        
        # Table selection
        tables = get_table_names(db_path)
        selected_table = st.selectbox(
            "Select Table",
            options=tables,
            help="Choose the table to train on"
        )
        
        if selected_table:
            # Get columns
            numeric_cols = get_numeric_columns(db_path, selected_table)
            
            # Target selection
            target_col = st.selectbox(
                "Select Target Column",
                options=numeric_cols,
                index=numeric_cols.index('Price') if 'Price' in numeric_cols else 0
            )
            
            # Model Selection
            st.markdown("#### 🤖 Model Selection")
            model_type = st.selectbox(
                "Select Model Type",
                options=['Auto ARIMA', 'ARIMA', 'SARIMA', 'Prophet'],
                help="Choose the type of time series model"
            )
            
            # Model Configuration
            st.markdown("#### ⚙️ Model Configuration")
            
            if model_type == 'Auto ARIMA':
                with st.expander("Auto ARIMA Configuration", expanded=True):
                    max_p = st.number_input('Maximum P (AR order)', 1, 10, 5)
                    max_d = st.number_input('Maximum D (Difference order)', 1, 5, 2)
                    max_q = st.number_input('Maximum Q (MA order)', 1, 10, 5)
                    use_seasonal = st.checkbox('Include Seasonal Components', value=True)
                    if use_seasonal:
                        seasonal_period = st.number_input('Seasonal Period', 1, 100, 5)
            
            elif model_type == 'ARIMA':
                p = st.number_input('P (AR order)', 0, 5, 1)
                d = st.number_input('D (Difference order)', 0, 2, 1)
                q = st.number_input('Q (MA order)', 0, 5, 1)
                order = (p, d, q)
            
            elif model_type == 'SARIMA':
                # Non-seasonal components
                p = st.number_input('P (AR order)', 0, 5, 1)
                d = st.number_input('D (Difference order)', 0, 2, 1)
                q = st.number_input('Q (MA order)', 0, 5, 1)
                
                # Seasonal components
                P = st.number_input('Seasonal P', 0, 5, 1)
                D = st.number_input('Seasonal D', 0, 2, 0)
                Q = st.number_input('Seasonal Q', 0, 5, 1)
                s = st.number_input('Seasonal Period', 1, 100, 5)
                
                order = (p, d, q)
                seasonal_order = (P, D, Q, s)
            
            elif model_type == 'Prophet':
                changepoint_prior_scale = st.slider(
                    'Changepoint Prior Scale',
                    0.001, 0.5, 0.05
                )
                seasonality_prior_scale = st.slider(
                    'Seasonality Prior Scale',
                    0.01, 10.0, 10.0
                )
            
            # Model name
            model_name = st.text_input(
                "Model Name",
                value=f"{model_type.lower()}_model_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
            
            # Training button
            if st.button("🚀 Train Model", type="primary"):
                try:
                    with st.spinner("Training model..."):
                        # Load data
                        conn = sqlite3.connect(db_path)
                        df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
                        conn.close()
                        
                        # Prepare time series
                        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                        df = df.set_index('DateTime')
                        series = df[target_col]
                        
                        if model_type == 'Auto ARIMA':
                            model, metrics = auto_arima(
                                series,
                                max_p=max_p,
                                max_d=max_d,
                                max_q=max_q,
                                seasonal=use_seasonal,
                                m=seasonal_period if use_seasonal else 1
                            )
                            
                        elif model_type == 'ARIMA':
                            model, metrics = train_arima(series, order)
                            
                        elif model_type == 'SARIMA':
                            model, metrics = train_sarima(series, order, seasonal_order)
                            
                        elif model_type == 'Prophet':
                            # Prepare data for Prophet
                            prophet_df = pd.DataFrame({
                                'ds': df.index,
                                'y': series
                            })
                            model, metrics = train_prophet(prophet_df)
                        
                        # Save model and metadata
                        model_path = os.path.join(models_dir, model_name)
                        os.makedirs(model_path, exist_ok=True)
                        
                        import joblib
                        joblib.dump(model, os.path.join(model_path, 'model.pkl'))
                        
                        metadata = {
                            'model_type': model_type,
                            'target_column': target_col,
                            'metrics': metrics,
                            'parameters': {
                                'order': order if model_type in ['ARIMA', 'SARIMA'] else None,
                                'seasonal_order': seasonal_order if model_type == 'SARIMA' else None,
                                'prophet_params': {
                                    'changepoint_prior_scale': changepoint_prior_scale,
                                    'seasonality_prior_scale': seasonality_prior_scale
                                } if model_type == 'Prophet' else None
                            }
                        }
                        
                        import json
                        with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
                            json.dump(metadata, f, indent=4)
                        
                        # Display results in right column
                        with right_col:
                            st.success(f"✨ Model trained successfully and saved to {model_path}")
                            
                            # Display metrics
                            st.markdown("#### 📈 Model Performance")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("MAE", f"{metrics['mae']:.4f}")
                                if 'aic' in metrics:
                                    st.metric("AIC", f"{metrics['aic']:.4f}")
                            with col2:
                                st.metric("RMSE", f"{metrics['rmse']:.4f}")
                                if 'bic' in metrics:
                                    st.metric("BIC", f"{metrics['bic']:.4f}")
                            
                            st.metric("R² Score", f"{metrics['r2']:.4f}")
                            
                            # Plot results
                            st.markdown("#### 📊 Model Fit")
                            if model_type == 'Prophet':
                                fig = model.plot(model.predict(prophet_df))
                                st.pyplot(fig)
                                
                                components_fig = model.plot_components(
                                    model.predict(prophet_df)
                                )
                                st.pyplot(components_fig)
                            else:
                                # Plot actual vs predicted
                                results_df = pd.DataFrame({
                                    'Actual': series,
                                    'Predicted': model.fittedvalues if model_type != 'Auto ARIMA' else model.predict(series)
                                })
                                st.line_chart(results_df)
                                
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    logging.error(f"Training error: {str(e)}", exc_info=True)
    
    # Right Column - Results
    with right_col:
        if not selected_table:
            st.info("👈 Please configure your model on the left to start training")

if __name__ == "__main__":
    setup_logging()
    time_series_page() 