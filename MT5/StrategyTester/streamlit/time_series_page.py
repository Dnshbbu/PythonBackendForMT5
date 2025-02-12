import streamlit as st
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
from db_info import get_table_names, get_numeric_columns
from feature_config import get_feature_groups, get_all_features
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    
    # Create main expander for metrics
    with right_col.expander("🎯 Model Performance", expanded=True):
        # Model Performance with enhanced visualization
        st.markdown("### 🏆 Model Performance")
        st.markdown("---")
        
        # Create three columns for metrics
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
               progress_bar=None,
               status_text=None) -> Tuple[object, Dict]:
    """
    Implement auto ARIMA using statsmodels with grid search
    
    Args:
        data: Time series data
        max_p: Maximum AR order
        max_d: Maximum difference order
        max_q: Maximum MA order
        seasonal: Whether to include seasonal components
        m: Seasonal period
        progress_bar: Streamlit progress bar placeholder
        status_text: Streamlit status text placeholder
        
    Returns:
        Tuple of (best model, metrics)
    """
    best_score = float('inf')
    best_model = None
    best_metrics = None
    best_order = None
    best_predictions = None
    
    # Create parameter grid
    p = range(0, max_p + 1)
    d = range(0, max_d + 1)
    q = range(0, max_q + 1)
    
    parameters = list(itertools.product(p, d, q))
    
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
            if progress_bar is not None:
                progress_bar.progress(progress)
            if status_text is not None:
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
                    best_predictions = results.fittedvalues
    
    # Clear progress indicators
    if progress_bar is not None:
        progress_bar.empty()
    if status_text is not None:
        status_text.empty()
    
    if best_model is None:
        raise ValueError("Could not find a suitable ARIMA model")
    
    st.info(f"Best ARIMA order found: {best_order}")
    
    # Store predictions for plotting but not in metrics
    best_model.predictions = best_predictions
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

def train_prophet(data: pd.DataFrame, features: List[str] = None) -> Dict:
    """Train Prophet model with optional additional regressors"""
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    
    # Add additional regressors if features are provided
    if features:
        for feature in features:
            model.add_regressor(feature)
    
    model.fit(data)
    
    # Make in-sample predictions
    future = model.make_future_dataframe(periods=0)
    if features:
        for feature in features:
            future[feature] = data[feature].values
    
    predictions = model.predict(future)
    
    # Calculate metrics
    metrics = {
        'mae': mean_absolute_error(data['y'], predictions['yhat']),
        'rmse': np.sqrt(mean_squared_error(data['y'], predictions['yhat'])),
        'r2': r2_score(data['y'], predictions['yhat'])
    }
    
    return model, metrics

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

def prepare_time_series_data(df: pd.DataFrame, target_col: str, selected_features: List[str], prediction_horizon: int = 1, n_lags: int = 3):
    """Prepare time series data for training to predict future prices
    
    Args:
        df: Input DataFrame
        target_col: Target column name (e.g., 'Price')
        selected_features: List of feature columns
        prediction_horizon: Number of steps ahead to predict (default: 1 for next row)
        n_lags: Number of previous price values to include as features (default: 3)
        
    Returns:
        Tuple of (features DataFrame, target Series) with target shifted for future prediction
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Add lagged price values as features
    for i in range(1, n_lags + 1):
        lag_col = f"{target_col}_lag_{i}"
        data[lag_col] = data[target_col].shift(i)
        if selected_features is not None:
            selected_features.append(lag_col)
    
    # Shift target column up by prediction_horizon to align current features with future target
    future_target = data[target_col].shift(-prediction_horizon)
    
    # Remove rows with NaN values due to lags at the beginning and prediction_horizon at the end
    data = data[n_lags:-prediction_horizon]
    future_target = future_target[n_lags:-prediction_horizon]
    
    # For Prophet, we need a specific format
    if selected_features:
        features = data[selected_features]
    else:
        features = None
        
    return data, future_target, features

def combine_tables_data(db_path: str, table_names: List[str]) -> pd.DataFrame:
    """Combine data from multiple tables into a single DataFrame
    
    Args:
        db_path: Path to the SQLite database
        table_names: List of table names to combine
        
    Returns:
        Combined DataFrame with data from all tables
    """
    combined_data = []
    
    try:
        conn = sqlite3.connect(db_path)
        
        for table_name in table_names:
            # Load data from each table
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            
            # Add table name as a source column
            df['DataSource'] = table_name
            
            # Convert date and time to datetime
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            
            combined_data.append(df)
            
            logging.info(f"Loaded {len(df)} rows from {table_name}")
        
        conn.close()
        
        # Combine all dataframes
        if not combined_data:
            raise ValueError("No data loaded from tables")
            
        combined_df = pd.concat(combined_data, axis=0, ignore_index=True)
        
        # Sort by DateTime
        combined_df = combined_df.sort_values('DateTime')
        
        logging.info(f"Combined data shape: {combined_df.shape}")
        return combined_df
        
    except Exception as e:
        logging.error(f"Error combining table data: {str(e)}")
        raise

def time_series_page():
    """Streamlit page for time series models"""
    # Initialize session state
    initialize_ts_session_state()
    
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
        
        def on_ts_table_selection_change():
            """Callback to handle table selection changes"""
            edited_rows = st.session_state['ts_table_editor']['edited_rows']
            for idx, changes in edited_rows.items():
                if '🔍 Select' in changes:
                    table_name = table_df.iloc[idx]['Table Name']
                    st.session_state['ts_table_selections'][table_name] = changes['🔍 Select']
            
            # Update selected tables list
            st.session_state['ts_selected_tables'] = [
                name for name, is_selected in st.session_state['ts_table_selections'].items() 
                if is_selected
            ]
        
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
            model_type = st.selectbox(
                "Select Model Type",
                options=['Auto ARIMA', 'ARIMA', 'SARIMA', 'Prophet'],
                help="Choose the type of time series model"
            )
            
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
                model_params['order'] = order
                if model_type == 'SARIMA':
                    model_params['seasonal_order'] = seasonal_order
            elif model_type == 'Prophet':
                model_params = {
                    'changepoint_prior_scale': changepoint_prior_scale,
                    'seasonality_prior_scale': seasonality_prior_scale
                }
            
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
                    
                    # Setup logging with Streamlit output
                    streamlit_handler = setup_ts_logging(training_progress)
                    
                    try:
                        with st.spinner("Training model..."):
                            # Load and combine data from all selected tables
                            logging.info(f"Loading data from {len(st.session_state['ts_selected_tables'])} tables")
                            df = combine_tables_data(db_path, st.session_state['ts_selected_tables'])
                            
                            # Check for stop
                            if check_ts_stop_clicked():
                                raise TrainingInterrupt("Training stopped by user")
                            
                            # Set DateTime as index
                            df = df.set_index('DateTime')
                            
                            # Prepare data for future prediction
                            data, future_target, features = prepare_time_series_data(
                                df, 
                                target_col, 
                                selected_features.copy(),  # Create a copy to avoid modifying original list
                                prediction_horizon=1,  # Predict next row's price
                                n_lags=n_lags if use_lagged_features else 0
                            )
                            
                            # Log the features being used
                            if features is not None:
                                logging.info(f"Using features: {features.columns.tolist()}")
                                logging.info(f"Number of lagged price features: {n_lags if use_lagged_features else 0}")
                            
                            # For Prophet, we need to prepare data differently
                            if model_type == 'Prophet':
                                prophet_df = pd.DataFrame({
                                    'ds': data.index,
                                    'y': future_target  # Use shifted target for future prediction
                                })
                                # Add selected features as regressors
                                if features is not None:
                                    for feature in selected_features:
                                        if feature != target_col:
                                            prophet_df[feature] = features[feature]
                                model, metrics = train_prophet(prophet_df, selected_features)
                            else:
                                # For ARIMA models, use the shifted target
                                series = future_target
                                exog = features
                                
                                if model_type == 'Auto ARIMA':
                                    model, metrics = auto_arima(
                                        series,
                                        max_p=max_p,
                                        max_d=max_d,
                                        max_q=max_q,
                                        seasonal=use_seasonal,
                                        m=seasonal_period if use_seasonal else 1,
                                        progress_bar=evaluation_progress_bar,
                                        status_text=evaluation_status
                                    )
                                elif model_type == 'ARIMA':
                                    model, metrics = train_arima(series, order)
                                elif model_type == 'SARIMA':
                                    model, metrics = train_sarima(series, order, seasonal_order)
                            
                            # Save model and metadata
                            model_path = os.path.join(models_dir, model_name)
                            os.makedirs(model_path, exist_ok=True)
                            
                            # Save model
                            import joblib
                            joblib.dump(model, os.path.join(model_path, 'model.pkl'))
                            
                            # Save metadata
                            metadata = {
                                'model_type': model_type,
                                'target_column': target_col,
                                'selected_features': selected_features,
                                'metrics': {
                                    key: float(value) if isinstance(value, (np.floating, np.integer)) 
                                    else value for key, value in metrics.items()
                                },
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
                            
                            # Display success message only if not stopped
                            if not check_ts_stop_clicked():
                                status_placeholder.success(f"✨ Model trained successfully and saved to {model_path}")
                                # Clear the stop button
                                stop_button_placeholder.empty()
                                
                                with metrics_placeholder:
                                    # Create tabs for different visualizations
                                    perf_tab, fit_tab = st.tabs(["📊 Model Performance", "📈 Model Fit"])
                                    
                                    # Model Performance Tab
                                    with perf_tab:
                                        display_ts_metrics(metrics, model_type, st)
                                    
                                    # Model Fit Tab
                                    with fit_tab:
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
                                                'Predicted': model.predictions if model_type == 'Auto ARIMA' else model.fittedvalues
                                            })
                                            st.line_chart(results_df)
                
                    except TrainingInterrupt:
                        if 'ts_stop_message' in st.session_state and st.session_state['ts_stop_message']:
                            status_placeholder.warning(st.session_state['ts_stop_message'])
                        # Clear the stop button
                        stop_button_placeholder.empty()
                        # Clean up any partial training artifacts if needed
                        if 'model_path' in locals():
                            try:
                                # Log cleanup attempt
                                logging.info(f"Cleaning up partial training artifacts in {model_path}")
                                # Add cleanup code here if needed
                            except Exception as cleanup_error:
                                logging.error(f"Error during cleanup: {str(cleanup_error)}")
                    
                    finally:
                        # Clean up logging handler
                        root_logger = logging.getLogger()
                        root_logger.removeHandler(streamlit_handler)
                        # Ensure stop button is removed in all cases
                        stop_button_placeholder.empty()
                        
                except Exception as e:
                    if not isinstance(e, TrainingInterrupt):
                        status_placeholder.error(f"Error during training: {str(e)}")
                        logging.error(f"Training error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    time_series_page() 