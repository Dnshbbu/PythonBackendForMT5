import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import sqlite3
import subprocess
from datetime import datetime
import logging
from db_info import get_table_names, get_numeric_columns
import json

def initialize_predict_ts_session_state():
    """Initialize session state variables for prediction page"""
    if 'predict_ts_selected_tables' not in st.session_state:
        st.session_state['predict_ts_selected_tables'] = []
    if 'predict_ts_table_selections' not in st.session_state:
        st.session_state['predict_ts_table_selections'] = {}
    if 'predict_ts_table_data' not in st.session_state:
        st.session_state['predict_ts_table_data'] = []
    if 'predict_ts_stop_clicked' not in st.session_state:
        st.session_state['predict_ts_stop_clicked'] = False
    if 'predict_ts_stop_message' not in st.session_state:
        st.session_state['predict_ts_stop_message'] = None
    if 'predict_ts_prediction_logs' not in st.session_state:
        st.session_state['predict_ts_prediction_logs'] = []
    if 'predict_ts_model_data' not in st.session_state:
        st.session_state['predict_ts_model_data'] = []
    if 'predict_ts_selected_model' not in st.session_state:
        st.session_state['predict_ts_selected_model'] = None
    if 'predict_ts_model_selections' not in st.session_state:
        st.session_state['predict_ts_model_selections'] = {}

def check_predict_ts_stop_clicked():
    """Check if stop button was clicked"""
    return st.session_state.get('predict_ts_stop_clicked', False)

def on_predict_ts_stop_click():
    """Callback for stop button click"""
    st.session_state['predict_ts_stop_clicked'] = True
    st.session_state['predict_ts_stop_message'] = "⚠️ Prediction was stopped by user"

def on_predict_ts_table_selection_change():
    """Callback to handle table selection changes"""
    edited_rows = st.session_state['predict_ts_table_editor']['edited_rows']
    
    # Get the table data from session state
    table_data = st.session_state['predict_ts_table_data']
    
    for idx, changes in edited_rows.items():
        if '🔍 Select' in changes:
            table_name = table_data[idx]['Table Name']
            st.session_state['predict_ts_table_selections'][table_name] = changes['🔍 Select']
    
    # Update selected tables list
    st.session_state['predict_ts_selected_tables'] = [
        name for name, is_selected in st.session_state['predict_ts_table_selections'].items() 
        if is_selected
    ]

def on_predict_ts_model_selection_change():
    """Callback to handle model selection changes"""
    edited_rows = st.session_state['predict_ts_model_editor']['edited_rows']
    
    # Get the model data from session state
    model_data = st.session_state['predict_ts_model_data']
    
    # Reset all selections first
    for model in model_data:
        model_name = model['Model Name']
        st.session_state['predict_ts_model_selections'][model_name] = False
    
    # Update the selected model
    for idx, changes in edited_rows.items():
        if '🔍 Select' in changes:
            model_name = model_data[idx]['Model Name']
            st.session_state['predict_ts_model_selections'][model_name] = changes['🔍 Select']
    
    # Update selected model (only one can be selected)
    selected_models = [
        name for name, is_selected in st.session_state['predict_ts_model_selections'].items() 
        if is_selected
    ]
    st.session_state['predict_ts_selected_model'] = selected_models[0] if selected_models else None

def get_available_tables_ts(db_path: str) -> List[Dict]:
    """Get list of available tables from the database with detailed information"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
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

def get_model_info(model_path: str) -> Dict:
    """Get model information from the model directory and name"""
    try:
        model_name = os.path.basename(model_path)
        
        # Extract model type from name (ts-arima, ts-prophet, ts-sarima)
        model_type = "Unknown"
        if "arima" in model_name.lower():
            model_type = "ARIMA" if "arima_" in model_name.lower() else "SARIMA"
        elif "prophet" in model_name.lower():
            model_type = "Prophet"
        elif "var" in model_name.lower():
            model_type = "VAR"
        
        # Try to read model_info.json if it exists
        info_path = os.path.join(model_path, 'model_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
                return {
                    'type': model_type,
                    'target': info.get('target', 'Price'),  # Default to Price if not specified
                    'features': len(info.get('features', [])),
                    'created': info.get('created_at', 
                        datetime.fromtimestamp(os.path.getctime(model_path)).strftime('%Y-%m-%d %H:%M:%S'))
                }
        
        # If no model_info.json, extract information from name and path
        created_time = datetime.fromtimestamp(os.path.getctime(model_path))
        
        # Extract training type (single, multi)
        training_type = "single" if "single" in model_name else "multiple"
        
        return {
            'type': model_type,
            'target': 'Price',  # Default target for time series models
            'features': 1,      # At least the target variable
            'created': created_time.strftime('%Y-%m-%d %H:%M:%S'),
            'training_type': training_type
        }
    except Exception as e:
        logging.error(f"Error getting model info: {str(e)}")
        return {
            'type': 'Unknown',
            'target': 'Price',
            'features': 1,
            'created': datetime.fromtimestamp(
                os.path.getctime(model_path)
            ).strftime('%Y-%m-%d %H:%M:%S')
        }

def get_available_models(models_dir: str) -> List[Dict]:
    """Get list of available trained models with detailed information"""
    try:
        models = []
        for model_name in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_name)
            if os.path.isdir(model_path):
                # Get model info
                info = get_model_info(model_path)
                models.append({
                    'name': model_name,
                    'type': info['type'],
                    'target': info['target'],
                    'features': info['features'],
                    'created': info['created']
                })
        return sorted(models, key=lambda x: x['created'], reverse=True)
    except Exception as e:
        st.error(f"Error accessing models directory: {str(e)}")
        return []

def display_predict_ts_evaluation_status():
    """Display prediction evaluation status"""
    status_container = st.container()
    
    with status_container:
        st.markdown("##### 🔄 Prediction Status")
        st.markdown("---")
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        return progress_bar, status_text

def get_predict_ts_equivalent_command(
    model_path: str,
    table_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_format: str = 'csv',
    forecast_horizon: int = 1,
    show_metrics: bool = True
) -> str:
    """Generate the equivalent command line command for time series prediction"""
    base_cmd = "python predict_time_series.py"
    model_arg = f"--model-path {model_path}"
    table_arg = f"--table {table_name}"
    
    # Add optional arguments
    optional_args = []
    if start_date:
        optional_args.append(f"--start-date {start_date}")
    if end_date:
        optional_args.append(f"--end-date {end_date}")
    if output_format:
        optional_args.append(f"--output-format {output_format}")
    if forecast_horizon != 1:
        optional_args.append(f"--forecast-horizon {forecast_horizon}")
    if show_metrics:
        optional_args.append("--show-metrics")
    
    # Combine all arguments
    optional_args_str = " ".join(optional_args)
    
    return f"{base_cmd} {model_arg} {table_arg} {optional_args_str}".strip()

def predict_time_series_page():
    """Streamlit page for time series predictions"""
    # Initialize session state
    initialize_predict_ts_session_state()
    
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
    models_dir = os.path.join(current_dir, 'models', 'time_series')
    
    # Create main left-right layout
    left_col, right_col = st.columns([1, 1], gap="large")
    
    # Create a single container for the right column that persists
    right_container = right_col.container()

    # Right Column - Results and Visualization
    with right_container:
        # Create tabs for organizing different sections
        status_tab, results_tab = st.tabs(["📊 Prediction Status", "📈 Results"])
        
        # Status Tab - Contains evaluation status and prediction progress
        with status_tab:
            # Prediction Status (always at top)
            evaluation_progress_bar, evaluation_status = display_predict_ts_evaluation_status()
            
            # Status message placeholder
            status_placeholder = st.empty()
            
            # Show stop message if exists
            if st.session_state.get('predict_ts_stop_message'):
                status_placeholder.warning(st.session_state['predict_ts_stop_message'])
            elif not st.session_state['predict_ts_selected_tables']:
                status_placeholder.info("👈 Please select a table and model on the left to start prediction.")
            
            # Stop button placeholder
            stop_button_placeholder = st.empty()
            
            # Prediction logs
            prediction_progress = st.empty()
            if st.session_state.get('predict_ts_prediction_logs'):
                with prediction_progress:
                    st.markdown("##### 📝 Prediction Progress")
                    st.code("\n".join(st.session_state['predict_ts_prediction_logs']))
        
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
        if not st.session_state['predict_ts_table_data']:
            table_data = []
            for t in available_tables:
                is_selected = st.session_state['predict_ts_table_selections'].get(t['name'], False)
                table_data.append({
                    '🔍 Select': is_selected,
                    'Table Name': t['name'],
                    'Date Range': t['date_range'],
                    'Rows': t['total_rows'],
                    'Symbols': ', '.join(t['symbols'])
                })
            st.session_state['predict_ts_table_data'] = table_data
        
        table_df = pd.DataFrame(st.session_state['predict_ts_table_data'])
        
        # Display table information with checkboxes
        edited_df = st.data_editor(
            table_df,
            hide_index=True,
            column_config={
                '🔍 Select': st.column_config.CheckboxColumn(
                    "Select",
                    help="Select table for prediction",
                    default=False
                )
            },
            key='predict_ts_table_editor',
            on_change=on_predict_ts_table_selection_change
        )

        # Model Selection Section
        if st.session_state['predict_ts_selected_tables']:
            st.markdown("#### 🤖 Model Selection")
            
            # Get available models with detailed information
            available_models = get_available_models(models_dir)
            
            if not available_models:
                st.warning("No trained models found. Please train a model first.")
                return
            
            # Create or use existing model data
            if not st.session_state['predict_ts_model_data']:
                model_data = []
                for m in available_models:
                    is_selected = st.session_state['predict_ts_model_selections'].get(m['name'], False)
                    model_data.append({
                        '🔍 Select': is_selected,
                        'Model Name': m['name'],
                        'Type': m['type'],
                        'Target': m['target'],
                        'Features': m['features'],
                        'Created': m['created']
                    })
                st.session_state['predict_ts_model_data'] = model_data
            
            model_df = pd.DataFrame(st.session_state['predict_ts_model_data'])
            
            # Display model information with radio buttons
            edited_df = st.data_editor(
                model_df,
                hide_index=True,
                column_config={
                    '🔍 Select': st.column_config.CheckboxColumn(
                        "Select",
                        help="Select model for prediction",
                        default=False
                    ),
                    'Model Name': st.column_config.TextColumn(
                        "Model Name",
                        help="Name of the trained model",
                        width="large"
                    ),
                    'Type': st.column_config.TextColumn(
                        "Model Type",
                        help="Type of the model (ARIMA, SARIMA, Prophet, VAR)",
                        width="medium"
                    ),
                    'Target': st.column_config.TextColumn(
                        "Target Variable",
                        help="Target variable used in training",
                        width="medium"
                    ),
                    'Features': st.column_config.NumberColumn(
                        "Features",
                        help="Number of features used in training",
                        width="small"
                    ),
                    'Created': st.column_config.TextColumn(
                        "Created At",
                        help="When the model was created",
                        width="medium"
                    )
                },
                disabled=["Model Name", "Type", "Target", "Features", "Created"],
                key='predict_ts_model_editor',
                on_change=on_predict_ts_model_selection_change,
                height=300  # Set a fixed height for better visibility
            )
            
            if not st.session_state['predict_ts_selected_model']:
                st.warning("Please select a model to proceed")
                return
            
            selected_model = st.session_state['predict_ts_selected_model']
            
            # Date Range Selection
            st.markdown("#### 📅 Date Range")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=None,
                    help="Start date for prediction (optional)"
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=None,
                    help="End date for prediction (optional)"
                )
            
            # Output Configuration
            st.markdown("#### 📤 Output Configuration")
            output_format = st.selectbox(
                "Output Format",
                options=['csv', 'json'],
                help="Choose the format for saving predictions"
            )
            
            # Advanced Options
            with st.expander("Advanced Options", expanded=False):
                forecast_horizon = st.number_input(
                    "Forecast Horizon",
                    min_value=1,
                    max_value=100,
                    value=1,
                    help="Number of future time points to predict"
                )
                show_metrics = st.checkbox(
                    "Show Model Metrics",
                    value=True,
                    help="Display model performance metrics"
                )
            
            # Display equivalent command
            st.markdown("##### 💻 Equivalent Command")
            cmd = get_predict_ts_equivalent_command(
                os.path.join(models_dir, selected_model),
                st.session_state['predict_ts_selected_tables'][0],
                start_date.strftime("%Y-%m-%d") if start_date else None,
                end_date.strftime("%Y-%m-%d") if end_date else None,
                output_format,
                forecast_horizon,
                show_metrics
            )
            st.code(cmd, language='bash')
            
            # Run Prediction button
            if st.button("🚀 Run Prediction", type="primary"):
                try:
                    # Clear right side content and reset session state
                    st.session_state['predict_ts_stop_clicked'] = False
                    st.session_state['predict_ts_prediction_logs'] = []
                    st.session_state['predict_ts_stop_message'] = None
                    
                    # Clear all placeholders
                    evaluation_progress_bar.empty()
                    evaluation_status.empty()
                    status_placeholder.empty()
                    prediction_progress.empty()
                    stop_button_placeholder.empty()
                    metrics_placeholder.empty()
                    
                    # Show stop button
                    with stop_button_placeholder:
                        st.button("🛑 Stop Prediction", 
                                on_click=on_predict_ts_stop_click,
                                key="predict_ts_stop_prediction",
                                help="Click to stop the prediction process",
                                type="secondary")
                    
                    # Execute the command
                    with st.spinner("Running prediction..."):
                        # Change directory to streamlit folder
                        os.chdir("C:\\Users\\StdUser\\Desktop\\MyProjects\\Backtesting\\MT5\\StrategyTester\\streamlit")
                        
                        # Start the process
                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            shell=True,
                            text=True,
                            bufsize=1,
                            universal_newlines=True
                        )
                        
                        # Create a single container for output
                        with prediction_progress:
                            st.markdown("##### 📝 Prediction Progress")
                            output_container = st.empty()
                            output_text = []
                            
                            # Progress tracking variables
                            total_steps = 6  # Load data, Init predictor, Prepare data, Predict, Format, Save
                            current_step = 0
                            step_progress = {
                                "Loading data": 0,
                                "Initializing predictor": 1,
                                "Preparing data": 2,
                                "Making predictions": 3,
                                "Formatting predictions": 4,
                                "Saving predictions": 5
                            }
                        
                        # Read output in real-time
                        while True:
                            line = process.stdout.readline()
                            if not line and process.poll() is not None:
                                break
                                
                            if line:
                                line = line.strip()
                                output_text.append(line)
                                
                                # Keep only last 1000 lines to prevent memory issues
                                if len(output_text) > 1000:
                                    output_text = output_text[-1000:]
                                
                                # Update the display with all output
                                output_container.code('\n'.join(output_text))
                                
                                # Update progress based on the log message
                                for step_name, step_num in step_progress.items():
                                    if step_name in line:
                                        current_step = step_num
                                        progress = (current_step + 1) / total_steps
                                        evaluation_progress_bar.progress(progress)
                                        evaluation_status.text(f"Step {current_step + 1}/{total_steps}: {line}")
                                        break
                                
                                # Check for completion
                                if "Prediction process completed successfully" in line:
                                    evaluation_progress_bar.progress(1.0)
                                    evaluation_status.text("Prediction completed successfully")
                                
                                # Check for errors
                                if "Error" in line:
                                    status_placeholder.error(line)
                            
                            # Check for stop button
                            if check_predict_ts_stop_clicked():
                                process.terminate()
                                status_placeholder.warning("Prediction stopped by user")
                                break
                        
                        # Wait for process to complete
                        process.wait()
                        
                        # Check return code
                        if process.returncode == 0:
                            if not check_predict_ts_stop_clicked():
                                status_placeholder.success("✨ Prediction completed successfully")
                                
                                # Try to load and display the predictions
                                try:
                                    # Get the latest prediction file
                                    predictions_dir = os.path.join(current_dir, 'predictions')
                                    prediction_files = [f for f in os.listdir(predictions_dir) 
                                                      if f.startswith('predictions_') and 
                                                      f.endswith(f'.{output_format}')]
                                    if prediction_files:
                                        latest_file = max(prediction_files, key=lambda x: os.path.getctime(
                                            os.path.join(predictions_dir, x)))
                                        
                                        # Load and display predictions in the Results tab
                                        with results_tab:
                                            st.markdown("#### 📊 Prediction Results")
                                            if output_format == 'csv':
                                                df = pd.read_csv(os.path.join(predictions_dir, latest_file))
                                                st.dataframe(df)
                                            else:  # json
                                                with open(os.path.join(predictions_dir, latest_file), 'r') as f:
                                                    predictions = json.load(f)
                                                st.json(predictions)
                                except Exception as e:
                                    st.warning(f"Could not load predictions: {str(e)}")
                        else:
                            status_placeholder.error("❌ Prediction failed")
                            evaluation_status.text("Prediction failed")
                            
                except Exception as e:
                    status_placeholder.error(f"Error during prediction: {str(e)}")
                    logging.error(f"Prediction error: {str(e)}", exc_info=True)
                finally:
                    # Ensure stop button is removed in all cases
                    stop_button_placeholder.empty()

if __name__ == "__main__":
    predict_time_series_page() 