import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import sqlite3
import subprocess
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from datetime import datetime
from db_info import get_table_names, get_numeric_columns
from feature_config import get_feature_groups, get_all_features
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from time_series_trainer import (
    auto_arima, train_arima, train_sarima, train_prophet, train_var,
    prepare_time_series_data, combine_tables_data, save_model,
    check_constant_series
)

class TrainingInterrupt(Exception):
    """Custom exception for interrupting the training process"""
    pass

def initialize_ts_session_state():
    """Initialize session state variables for time series page"""
    if 'ts_selected_tables' not in st.session_state:
        st.session_state['ts_selected_tables'] = []
    if 'ts_table_selections' not in st.session_state:
        st.session_state['ts_table_selections'] = {}
    if 'ts_table_data' not in st.session_state:
        st.session_state['ts_table_data'] = []
    if 'ts_stop_clicked' not in st.session_state:
        st.session_state['ts_stop_clicked'] = False
    if 'ts_stop_message' not in st.session_state:
        st.session_state['ts_stop_message'] = None
    if 'ts_training_logs' not in st.session_state:
        st.session_state['ts_training_logs'] = []
    if 'ts_previous_selection' not in st.session_state:
        st.session_state['ts_previous_selection'] = {
            'tables': [],
            'model_type': None
        }

def check_ts_stop_clicked():
    """Check if stop button was clicked"""
    return st.session_state.get('ts_stop_clicked', False)

def on_ts_stop_click():
    """Callback for stop button click"""
    st.session_state['ts_stop_clicked'] = True
    st.session_state['ts_stop_message'] = "⚠️ Training was stopped by user"

class StreamlitTSHandler(logging.Handler):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        if 'ts_training_logs' not in st.session_state:
            st.session_state['ts_training_logs'] = []
        self.logs = st.session_state['ts_training_logs']
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.logs.append(msg)
            
            # Keep only last 1000 messages
            if len(self.logs) > 1000:
                self.logs = self.logs[-1000:]
            
            # Update the placeholder with all logs
            log_text = "\n".join(self.logs)
            self.placeholder.code(log_text)
            
            # Force a Streamlit rerun to update the UI
            st.session_state['ts_training_logs'] = self.logs
            
            # Check for stop after each log message
            if check_ts_stop_clicked():
                raise TrainingInterrupt("Training stopped by user")
            
        except TrainingInterrupt:
            raise
        except Exception:
            self.handleError(record)

def setup_ts_logging(placeholder):
    """Configure logging settings with Streamlit output"""
    # Create Streamlit handler
    streamlit_handler = StreamlitTSHandler(placeholder)
    streamlit_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    
    # Get root logger and add handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our custom handler
    root_logger.addHandler(streamlit_handler)
    
    return streamlit_handler

def display_ts_metrics(metrics: Dict, model_type: str, right_col):
    """Display training metrics in a formatted way"""
    if not metrics:
        return
    
    # Model Performance with enhanced visualization
    st.markdown(f"#### 🎯 Model Performance - {model_type}")
    st.markdown("---")
    
    if model_type == 'VAR':
        # For VAR models, show metrics for each variable
        st.markdown("#### Overall Model Information")
        col1, col2 = st.columns(2)
        col1.metric("Model Type", model_type)
        col1.metric("Number of Variables", len([k for k in metrics.keys() if k.endswith('_mae')]))
        col2.metric("Model Order", metrics.get('order', 'N/A'))
        col2.metric("Number of Observations", metrics.get('n_observations', 'N/A'))
        
        # Show metrics for each variable
        st.markdown("#### Variable-wise Metrics")
        st.markdown("---")
        
        # Create a container for variable metrics
        var_metrics_container = st.container()
        
        with var_metrics_container:
            for col in [k.replace('_mae', '') for k in metrics.keys() if k.endswith('_mae')]:
                st.markdown(f"**Metrics for {col}**")
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "MAE",
                    f"{metrics.get(f'{col}_mae', 0):.4f}",
                    help="Mean Absolute Error"
                )
                col2.metric(
                    "RMSE",
                    f"{metrics.get(f'{col}_rmse', 0):.4f}",
                    help="Root Mean Square Error"
                )
                col3.metric(
                    "R²",
                    f"{metrics.get(f'{col}_r2', 0):.4f}",
                    help="R-squared score"
                )
                st.markdown("---")
    else:
        # Original metrics display for other models
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Model Type",
                model_type,
                delta=None,
                help="The trained model type"
            )
            st.metric(
                "MAE",
                f"{metrics.get('mae', 0):.4f}",
                delta=None,
                help="Mean Absolute Error"
            )
        
        with col2:
            st.metric(
                "RMSE",
                f"{metrics.get('rmse', 0):.4f}",
                delta=None,
                help="Root Mean Square Error"
            )
            st.metric(
                "R²",
                f"{metrics.get('r2', 0):.4f}",
                delta=None,
                help="R-squared score"
            )
        
        with col3:
            if 'aic' in metrics:
                st.metric(
                    "AIC",
                    f"{metrics.get('aic', 0):.4f}",
                    delta=None,
                    help="Akaike Information Criterion"
                )
            if 'bic' in metrics:
                st.metric(
                    "BIC",
                    f"{metrics.get('bic', 0):.4f}",
                    delta=None,
                    help="Bayesian Information Criterion"
                )

def get_available_tables_ts(db_path: str) -> List[Dict]:
    """Get list of available tables from the database with detailed information"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get tables with their creation time from sqlite_master
        cursor.execute("""
            SELECT name 
            FROM sqlite_master 
            WHERE type='table' AND name LIKE 'strategy_%'
            ORDER BY name DESC
        """)
        
        tables = []
        for (table_name,) in cursor.fetchall():
            try:
                # Get date range
                cursor.execute(f'SELECT MIN(Date || " " || Time), MAX(Date || " " || Time) FROM {table_name}')
                start_time, end_time = cursor.fetchone()
                
                # Get row count
                cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
                row_count = cursor.fetchone()[0]
                
                # Get unique symbols
                cursor.execute(f'SELECT DISTINCT Symbol FROM {table_name}')
                symbols = [row[0] for row in cursor.fetchall()]
                
                tables.append({
                    'name': table_name,
                    'date_range': f"{start_time} to {end_time}",
                    'total_rows': row_count,
                    'symbols': symbols
                })
                
            except Exception as e:
                logging.error(f"Error getting details for table {table_name}: {str(e)}")
                continue
        
        conn.close()
        return tables
    
    except Exception as e:
        st.error(f"Error accessing database: {str(e)}")
        return []

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_ts_equivalent_command(
    table_names: List[str], 
    target_col: str,
    selected_features: List[str],
    model_type: str,
    model_name: str,
    **model_params
) -> str:
    """Generate the equivalent command line command for time series models"""
    base_cmd = "python train_time_series_models.py"
    tables_arg = f"--tables {' '.join(table_names)}"
    target_arg = f"--target {target_col}"
    features_arg = f"--features {' '.join(selected_features)}"
    model_type_arg = f"--model-type {model_type}"
    model_name_arg = f"--model-name {model_name}"
    
    # Add model-specific parameters
    params_list = []
    if model_type == 'Auto ARIMA':
        params_list.extend([
            f"--max-p {model_params.get('max_p', 5)}",
            f"--max-d {model_params.get('max_d', 2)}",
            f"--max-q {model_params.get('max_q', 5)}",
            f"--seasonal {str(model_params.get('use_seasonal', True)).lower()}"
        ])
        if model_params.get('use_seasonal', True):
            params_list.append(f"--seasonal-period {model_params.get('seasonal_period', 5)}")
    
    elif model_type == 'ARIMA':
        order = model_params.get('order', (1,1,1))
        params_list.append(f"--order {order[0]} {order[1]} {order[2]}")
    
    elif model_type == 'SARIMA':
        order = model_params.get('order', (1,1,1))
        seasonal_order = model_params.get('seasonal_order', (1,0,1,5))
        params_list.extend([
            f"--order {order[0]} {order[1]} {order[2]}",
            f"--seasonal-order {seasonal_order[0]} {seasonal_order[1]} {seasonal_order[2]} {seasonal_order[3]}"
        ])
    
    elif model_type == 'Prophet':
        params_list.extend([
            f"--changepoint-prior-scale {model_params.get('changepoint_prior_scale', 0.05)}",
            f"--seasonality-prior-scale {model_params.get('seasonality_prior_scale', 10.0)}"
        ])
    
    elif model_type == 'VAR':
        params_list.extend([
            f"--maxlags {model_params.get('maxlags', 5)}"
        ])
    
    params_str = " ".join(params_list)
    return f"{base_cmd} {tables_arg} {target_arg} {features_arg} {model_type_arg} {model_name_arg} {params_str}".strip()

def display_ts_evaluation_status():
    """Display model evaluation status"""
    # Create container for evaluation status
    status_container = st.container()
    
    with status_container:
        st.markdown("##### 🔄 Model Evaluation Status")
        st.markdown("---")
        # Progress bar placeholder
        progress_bar = st.empty()
        # Status text placeholder
        status_text = st.empty()
        return progress_bar, status_text

def generate_model_name(model_type: str, training_type: str, timestamp: Optional[str] = None) -> str:
    """Generate consistent model name
    
    Args:
        model_type: Type of model (e.g., 'xgboost', 'decision_tree')
        training_type: Type of training ('single', 'multi', 'incremental', 'base')
        timestamp: Optional timestamp, will generate if None
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"ts-{model_type}_{training_type}_{timestamp}"

def clear_training_outputs():
    """Clear all training outputs and logs"""
    st.session_state['ts_training_logs'] = []
    st.session_state['ts_stop_clicked'] = False
    st.session_state['ts_stop_message'] = None

def on_ts_table_selection_change():
    """Callback to handle table selection changes"""
    edited_rows = st.session_state['ts_table_editor']['edited_rows']
    current_tables = []
    
    # Get the table data from session state
    table_data = st.session_state['ts_table_data']
    
    for idx, changes in edited_rows.items():
        if '🔍 Select' in changes:
            # Use the table data from session state instead of table_df
            table_name = table_data[idx]['Table Name']
            st.session_state['ts_table_selections'][table_name] = changes['🔍 Select']
            if changes['🔍 Select']:
                current_tables.append(table_name)
    
    # Update selected tables list
    st.session_state['ts_selected_tables'] = [
        name for name, is_selected in st.session_state['ts_table_selections'].items() 
        if is_selected
    ]
    
    # Clear outputs if table selection changed
    if set(current_tables) != set(st.session_state['ts_previous_selection']['tables']):
        clear_training_outputs()
        st.session_state['ts_previous_selection']['tables'] = current_tables.copy()

def on_model_type_change():
    """Callback for model type change"""
    # Get the current model type from session state
    current_model = st.session_state['model_type_selector']
    
    # Clear outputs if model type changed
    if current_model != st.session_state['ts_previous_selection']['model_type']:
        clear_training_outputs()
        st.session_state['ts_previous_selection']['model_type'] = current_model

def time_series_page():
    """Streamlit page for time series models"""
    # Initialize session state
    initialize_ts_session_state()
    
    # st.markdown("""
    #     <h2 style='text-align: center;'>📈 Time Series Models</h2>
    #     <p style='text-align: center; color: #666;'>
    #         Train specialized time series models for prediction
    #     </p>
    #     <hr>
    # """, unsafe_allow_html=True)
    
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
    models_dir = os.path.join(current_dir, 'models', 'time_series')
    os.makedirs(models_dir, exist_ok=True)
    
    # Create main left-right layout
    left_col, right_col = st.columns([1, 1], gap="large")
    
    # Create a single container for the right column that persists
    right_container = right_col.container()

    # Right Column - Results and Visualization
    with right_container:
        # Create tabs for organizing different sections
        status_tab, results_tab = st.tabs(["📊 Model Status", "📈 Results"])
        
        # Status Tab - Contains evaluation status and training progress
        with status_tab:
            # Model Evaluation Status (always at top)
            evaluation_progress_bar, evaluation_status = display_ts_evaluation_status()
            
            # Define progress callback here where we have access to the progress elements
            def progress_callback(progress, message):
                """Callback function to update progress bar and status"""
                evaluation_progress_bar.progress(progress)
                evaluation_status.text(message)
            
            # Status message placeholder
            status_placeholder = st.empty()
            
            # Show stop message if exists
            if st.session_state.get('ts_stop_message'):
                status_placeholder.warning(st.session_state['ts_stop_message'])
            elif not st.session_state['ts_selected_tables']:
                status_placeholder.info("👈 Please select tables and configure your model on the left to start training.")
            
            # Stop button placeholder
            stop_button_placeholder = st.empty()
            
            # Training logs
            training_progress = st.empty()
            if st.session_state.get('ts_training_logs'):
                with training_progress:
                    st.markdown("##### 📝 Training Progress")
                    st.code("\n".join(st.session_state['ts_training_logs']))
        
        # Results Tab - Contains metrics and visualizations
        with results_tab:
            metrics_placeholder = st.empty()

    # Left Column - Configuration
    with left_col:
        st.markdown("#### 📊 Data Configuration")
        
        # Get available tables with detailed information
        available_tables = get_available_tables_ts(db_path)
        
        if not available_tables:
            st.warning("No strategy tables found in the database.")
            return
        
        # Create or use existing table data
        if not st.session_state['ts_table_data']:
            table_data = []
            for t in available_tables:
                # Use the stored selection state or default to False
                is_selected = st.session_state['ts_table_selections'].get(t['name'], False)
                table_data.append({
                    '🔍 Select': is_selected,
                    'Table Name': t['name'],
                    'Date Range': t['date_range'],
                    'Rows': t['total_rows'],
                    'Symbols': ', '.join(t['symbols'])
                })
            st.session_state['ts_table_data'] = table_data
        
        table_df = pd.DataFrame(st.session_state['ts_table_data'])
        
        # Display table information with checkboxes
        edited_df = st.data_editor(
            table_df,
            hide_index=True,
            column_config={
                '🔍 Select': st.column_config.CheckboxColumn(
                    "Select",
                    help="Select tables for training",
                    default=False
                )
            },
            key='ts_table_editor',
            on_change=on_ts_table_selection_change
        )

        # Model Configuration Section
        if st.session_state['ts_selected_tables']:
            # Get numeric columns for the first selected table
            first_table = st.session_state['ts_selected_tables'][0]
            numeric_cols = get_numeric_columns(db_path, first_table)
            
            # Target column selection
            target_col = st.selectbox(
                "Select Target Column",
                options=numeric_cols,
                index=numeric_cols.index('Price') if 'Price' in numeric_cols else 0
            )
            
            # Feature Selection Section
            st.markdown("##### 🎨 Feature Selection")
            
            # Get feature groups
            feature_groups = get_feature_groups()
            all_features = get_all_features()
            
            # Feature group selection
            st.markdown("**Select Feature Groups:**")
            selected_groups = {}
            for group_name, group_features in feature_groups.items():
                selected_groups[group_name] = st.checkbox(
                    f"{group_name.title()} Features",
                    value=True,
                    help=f"Select all {group_name} features"
                )
            
            # Individual feature selection
            st.markdown("**Fine-tune Feature Selection:**")
            selected_features = []
            for group_name, group_features in feature_groups.items():
                if selected_groups[group_name]:
                    with st.expander(f"{group_name.title()} Features"):
                        for feature in group_features:
                            if feature in numeric_cols and st.checkbox(
                                feature,
                                value=True,
                                key=f"ts_feature_{feature}"
                            ):
                                selected_features.append(feature)
            
            # Additional numeric columns not in predefined groups
            other_numeric_cols = [col for col in numeric_cols if col not in all_features and col != target_col]
            if other_numeric_cols:
                with st.expander("Additional Numeric Features"):
                    for col in other_numeric_cols:
                        if st.checkbox(col, value=False, key=f"ts_feature_{col}"):
                            selected_features.append(col)
            
            # Model Configuration
            st.markdown("#### ⚙️ Model Configuration")
            
            # Model Selection
            st.markdown("#### 🤖 Model Selection")
            training_mode = st.selectbox(
                "Training Mode",
                options=["Single Model", "Multiple Models"],
                key='model_type_selector',
                on_change=on_model_type_change
            )
            
            if training_mode == "Multiple Models":
                st.info("🤖 Select which models to include in training")
                
                # Option to select all models
                use_all_models = st.checkbox("Use All Available Models", value=True, 
                                           help="Select this to use all available models")
                
                available_models = ['Auto ARIMA', 'ARIMA', 'SARIMA', 'Prophet', 'VAR']
                
                # If not using all models, show multi-select
                if not use_all_models:
                    selected_models = st.multiselect(
                        "Select Models to Include",
                        options=available_models,
                        default=['Auto ARIMA', 'Prophet'],
                        help="Choose which models to include in the training process"
                    )
                    if not selected_models:
                        st.warning("⚠️ Please select at least one model")
                else:
                    selected_models = available_models
                
                # Model Configuration for each selected model
                model_configs = {}
                for model_type in selected_models:
                    with st.expander(f"{model_type} Configuration", expanded=False):
                        if model_type == 'Auto ARIMA':
                            max_p = st.number_input(f'{model_type} - Maximum P (AR order)', 1, 10, 5)
                            max_d = st.number_input(f'{model_type} - Maximum D (Difference order)', 1, 5, 2)
                            max_q = st.number_input(f'{model_type} - Maximum Q (MA order)', 1, 10, 5)
                            use_seasonal = st.checkbox(f'{model_type} - Include Seasonal Components', value=True)
                            seasonal_period = st.number_input(f'{model_type} - Seasonal Period', 1, 100, 5) if use_seasonal else None
                            model_configs[model_type] = {
                                'max_p': max_p,
                                'max_d': max_d,
                                'max_q': max_q,
                                'use_seasonal': use_seasonal,
                                'seasonal_period': seasonal_period
                            }
                        
                        elif model_type == 'ARIMA':
                            p = st.number_input(f'{model_type} - P (AR order)', 0, 5, 1)
                            d = st.number_input(f'{model_type} - D (Difference order)', 0, 2, 1)
                            q = st.number_input(f'{model_type} - Q (MA order)', 0, 5, 1)
                            model_configs[model_type] = {
                                'order': (p, d, q)
                            }
                        
                        elif model_type == 'SARIMA':
                            p = st.number_input(f'{model_type} - P (AR order)', 0, 5, 1)
                            d = st.number_input(f'{model_type} - D (Difference order)', 0, 2, 1)
                            q = st.number_input(f'{model_type} - Q (MA order)', 0, 5, 1)
                            P = st.number_input(f'{model_type} - Seasonal P', 0, 5, 1)
                            D = st.number_input(f'{model_type} - Seasonal D', 0, 2, 0)
                            Q = st.number_input(f'{model_type} - Seasonal Q', 0, 5, 1)
                            s = st.number_input(f'{model_type} - Seasonal Period', 1, 100, 5)
                            model_configs[model_type] = {
                                'order': (p, d, q),
                                'seasonal_order': (P, D, Q, s)
                            }
                        
                        elif model_type == 'Prophet':
                            changepoint_prior_scale = st.slider(
                                f'{model_type} - Changepoint Prior Scale',
                                0.001, 0.5, 0.05
                            )
                            seasonality_prior_scale = st.slider(
                                f'{model_type} - Seasonality Prior Scale',
                                0.01, 10.0, 10.0
                            )
                            model_configs[model_type] = {
                                'changepoint_prior_scale': changepoint_prior_scale,
                                'seasonality_prior_scale': seasonality_prior_scale
                            }
                        
                        elif model_type == 'VAR':
                            maxlags = st.number_input(f'{model_type} - Maximum Lags', 1, 20, 5)
                            model_configs[model_type] = {
                                'maxlags': maxlags
                            }
                
                model_type = "multiple"  # Set for model name generation
            else:
                model_type = st.selectbox(
                    "Select Model Type",
                    options=['Auto ARIMA', 'ARIMA', 'SARIMA', 'Prophet', 'VAR'],
                    help="Choose the type of time series model",
                    key='single_model_selector'
                )
                
                # Model Configuration
                if model_type == 'Auto ARIMA':
                    with st.expander("Auto ARIMA Configuration", expanded=True):
                        max_p = st.number_input('Maximum P (AR order)', 1, 10, 5)
                        max_d = st.number_input('Maximum D (Difference order)', 1, 5, 2)
                        max_q = st.number_input('Maximum Q (MA order)', 1, 10, 5)
                        use_seasonal = st.checkbox('Include Seasonal Components', value=True)
                        seasonal_period = st.number_input('Seasonal Period', 1, 100, 5) if use_seasonal else None
                        model_configs = {
                            model_type: {
                                'max_p': max_p,
                                'max_d': max_d,
                                'max_q': max_q,
                                'use_seasonal': use_seasonal,
                                'seasonal_period': seasonal_period
                            }
                        }
                
                elif model_type == 'ARIMA':
                    p = st.number_input('P (AR order)', 0, 5, 1)
                    d = st.number_input('D (Difference order)', 0, 2, 1)
                    q = st.number_input('Q (MA order)', 0, 5, 1)
                    model_configs = {
                        model_type: {
                            'order': (p, d, q)
                        }
                    }
                
                elif model_type == 'SARIMA':
                    p = st.number_input('P (AR order)', 0, 5, 1)
                    d = st.number_input('D (Difference order)', 0, 2, 1)
                    q = st.number_input('Q (MA order)', 0, 5, 1)
                    P = st.number_input('Seasonal P', 0, 5, 1)
                    D = st.number_input('Seasonal D', 0, 2, 0)
                    Q = st.number_input('Seasonal Q', 0, 5, 1)
                    s = st.number_input('Seasonal Period', 1, 100, 5)
                    model_configs = {
                        model_type: {
                            'order': (p, d, q),
                            'seasonal_order': (P, D, Q, s)
                        }
                    }
                
                elif model_type == 'Prophet':
                    changepoint_prior_scale = st.slider(
                        'Changepoint Prior Scale',
                        0.001, 0.5, 0.05
                    )
                    seasonality_prior_scale = st.slider(
                        'Seasonality Prior Scale',
                        0.01, 10.0, 10.0
                    )
                    model_configs = {
                        model_type: {
                            'changepoint_prior_scale': changepoint_prior_scale,
                            'seasonality_prior_scale': seasonality_prior_scale
                        }
                    }
                
                elif model_type == 'VAR':
                    with st.expander("VAR Configuration", expanded=True):
                        maxlags = st.number_input('Maximum Lags', 1, 20, 5)
                        model_configs = {
                            model_type: {
                                'maxlags': maxlags
                            }
                        }
                
                selected_models = [model_type]
            
            # Model name - auto-generated based on model type and timestamp
            model_name = generate_model_name(
                model_type=model_type.lower(),
                training_type="single",
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            
            # Display equivalent command
            st.markdown("##### 💻 Equivalent Command")
            
            # Prepare model parameters
            model_params = {}
            if model_type == 'Auto ARIMA':
                model_params = {
                    'max_p': max_p,
                    'max_d': max_d,
                    'max_q': max_q,
                    'use_seasonal': use_seasonal
                }
                if use_seasonal:
                    model_params['seasonal_period'] = seasonal_period
            elif model_type in ['ARIMA', 'SARIMA']:
                model_params['order'] = model_configs[model_type]['order']
            elif model_type == 'Prophet':
                model_params = model_configs[model_type]
            elif model_type == 'VAR':
                model_params = model_configs[model_type]
            
            cmd = get_ts_equivalent_command(
                st.session_state['ts_selected_tables'],
                target_col,
                selected_features,
                model_type,
                model_name,
                **model_params
            )
            st.code(cmd, language='bash')
            
            # Add configuration for lagged features
            st.markdown("#### 🕒 Lagged Features Configuration")
            use_lagged_features = st.checkbox("Use Previous Price Values as Features", value=True,
                                            help="Include previous price values to improve prediction")
            if use_lagged_features:
                n_lags = st.number_input("Number of Previous Prices to Use", 
                                       min_value=1, max_value=10, value=3,
                                       help="Number of previous price values to include as features")
            else:
                n_lags = 0
            
            # Training button
            if st.button("🚀 Train Model", type="primary"):
                try:
                    # Clear right side content and reset session state
                    st.session_state['ts_stop_clicked'] = False
                    st.session_state['ts_training_logs'] = []
                    st.session_state['ts_stop_message'] = None
                    
                    # Clear all placeholders
                    evaluation_progress_bar.empty()
                    evaluation_status.empty()
                    status_placeholder.empty()
                    training_progress.empty()
                    stop_button_placeholder.empty()
                    metrics_placeholder.empty()
                    
                    # Show stop button
                    with stop_button_placeholder:
                        st.button("🛑 Stop Training", 
                                on_click=on_ts_stop_click,
                                key="ts_stop_training",
                                help="Click to stop the training process",
                                type="secondary")
                    
                    # Get the command to execute
                    cmd = get_ts_equivalent_command(
                        st.session_state['ts_selected_tables'],
                                target_col, 
                        selected_features,
                        model_type,
                        model_name,
                        **model_params
                    )
                    
                    # Execute the command
                    with st.spinner("Training model..."):
                        # Change directory to streamlit folder
                        os.chdir("C:\\Users\\StdUser\\Desktop\\MyProjects\\Backtesting\\MT5\\StrategyTester\\streamlit")
                        
                        # Start the process
                        process = subprocess.Popen(
                            cmd,  # Use full command string
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            shell=True,  # Use shell=True to handle command string
                            text=True,
                            bufsize=1,
                            universal_newlines=True
                        )
                        
                        # Create a single container for output
                        with training_progress:
                            st.markdown("##### 📝 Training Progress")
                            output_container = st.empty()
                            output_text = []
                        
                        # Read output in real-time
                        while True:
                            line = process.stdout.readline()
                            if not line and process.poll() is not None:
                                break
                                
                            if line:
                                output_text.append(line.strip())
                                # Keep only last 1000 lines to prevent memory issues
                                if len(output_text) > 1000:
                                    output_text = output_text[-1000:]
                                
                                # Update the display with all output
                                output_container.code('\n'.join(output_text))
                                
                                # Update progress bar if percentage is found
                                if "%" in line:
                                    try:
                                        progress = float(line.split("%")[0].strip().split()[-1]) / 100
                                        evaluation_progress_bar.progress(progress)
                                        evaluation_status.text(line.strip())
                                    except:
                                        pass
                            
                            # Check for stop button
                            if check_ts_stop_clicked():
                                process.terminate()
                                status_placeholder.warning("Training stopped by user")
                                break
                        
                        # Wait for process to complete
                        process.wait()
                        
                        # Check return code
                        if process.returncode == 0:
                            if check_ts_stop_clicked():
                                status_placeholder.warning("Training stopped by user")
                            else:
                                status_placeholder.success("✨ Training completed successfully")
                                evaluation_progress_bar.progress(1.0)
                                evaluation_status.text("Training completed successfully")
                                
                                # Display model paths if found in logs
                                model_path = None
                                scaler_path = None
                                for line in output_text:
                                    if "Model saved to" in line:
                                        model_path = line.split("Model saved to")[-1].strip()
                                    elif "Scaler saved to" in line:
                                        scaler_path = line.split("Scaler saved to")[-1].strip()
                                
                                if model_path or scaler_path:
                                    with st.expander("📁 Model Files", expanded=True):
                                        if model_path:
                                            st.markdown(f"**Model Path**: `{model_path}`")
                                        if scaler_path:
                                            st.markdown(f"**Scaler Path**: `{scaler_path}`")
                        else:
                            status_placeholder.error("❌ Training failed")
                            evaluation_status.text("Training failed")
                            
                except Exception as e:
                    status_placeholder.error(f"Error during training: {str(e)}")
                    logging.error(f"Training error: {str(e)}", exc_info=True)
                finally:
                    # Ensure stop button is removed in all cases
                    stop_button_placeholder.empty()

if __name__ == "__main__":
    time_series_page() 