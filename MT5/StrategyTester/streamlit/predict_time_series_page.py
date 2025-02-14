import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
import subprocess
import sqlite3
from typing import List, Dict
from time_series_predictor import (
    load_model,
    prepare_data_for_prediction,
    make_predictions,
    get_model_info
)

def initialize_ts_pred_session_state():
    """Initialize session state variables for time series prediction page"""
    if 'ts_pred_model_path' not in st.session_state:
        st.session_state['ts_pred_model_path'] = None
    if 'ts_pred_model_info' not in st.session_state:
        st.session_state['ts_pred_model_info'] = None
    if 'ts_pred_data' not in st.session_state:
        st.session_state['ts_pred_data'] = None
    if 'ts_pred_stop_clicked' not in st.session_state:
        st.session_state['ts_pred_stop_clicked'] = False
    if 'ts_pred_stop_message' not in st.session_state:
        st.session_state['ts_pred_stop_message'] = None
    if 'ts_pred_selected_table' not in st.session_state:
        st.session_state['ts_pred_selected_table'] = None
    if 'ts_pred_selected_model' not in st.session_state:
        st.session_state['ts_pred_selected_model'] = None
    if 'ts_pred_model_selections' not in st.session_state:
        st.session_state['ts_pred_model_selections'] = {}
    if 'ts_pred_model_data' not in st.session_state:
        st.session_state['ts_pred_model_data'] = []
    if 'ts_pred_table_selections' not in st.session_state:
        st.session_state['ts_pred_table_selections'] = {}
    if 'ts_pred_table_data' not in st.session_state:
        st.session_state['ts_pred_table_data'] = []

def get_available_tables(db_path: str) -> List[Dict]:
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
                
                # Create a display name that includes key information
                display_name = f"{table_name} ({start_time} to {end_time}, {row_count} rows)"
                
                tables.append({
                    'name': table_name,
                    'date_range': f"{start_time} to {end_time}",
                    'total_rows': row_count,
                    'symbols': symbols,
                    'display_name': display_name
                })
                
            except Exception as e:
                logging.error(f"Error getting details for table {table_name}: {str(e)}")
                continue
        
        conn.close()
        return tables
    
    except Exception as e:
        st.error(f"Error accessing database: {str(e)}")
        return []

def check_ts_pred_stop_clicked():
    """Check if stop button was clicked"""
    return st.session_state.get('ts_pred_stop_clicked', False)

def on_ts_pred_stop_click():
    """Callback for stop button click"""
    st.session_state['ts_pred_stop_clicked'] = True
    st.session_state['ts_pred_stop_message'] = "⚠️ Prediction was stopped by user"

def display_model_info(model_info: dict):
    """Display model information in a formatted way"""
    st.markdown("### 📊 Model Information")
    
    # Create columns for different types of information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Model Details")
        st.write(f"**Model Type:** {model_info.get('model_type', 'N/A')}")
        st.write(f"**Training Date:** {model_info.get('training_date', 'N/A')}")
        
        # Display model parameters if available
        if 'model_params' in model_info:
            st.markdown("#### ⚙️ Model Parameters")
            for param, value in model_info['model_params'].items():
                st.write(f"**{param}:** {value}")
    
    with col2:
        st.markdown("#### 📈 Features")
        if 'features' in model_info:
            st.write("**Required Features:**")
            for feature in model_info['features']:
                st.write(f"- {feature}")
        
        if 'target' in model_info:
            st.write(f"**Target Column:** {model_info['target']}")

def get_pred_equivalent_command(
    model_path: str,
    table_name: str,
    output_path: str = None,
    start_date: str = None,
    end_date: str = None,
    output_format: str = 'csv',
    show_metrics: bool = True,
    forecast_horizon: int = 1
) -> str:
    """Generate the equivalent command for prediction"""
    base_cmd = "python predict_time_series.py"
    model_arg = f"--model-path {model_path}"
    table_arg = f"--table {table_name}"
    
    args = [base_cmd, model_arg, table_arg]
    
    if output_path:
        args.append(f"--output-path {output_path}")
    if start_date:
        args.append(f"--start-date {start_date}")
    if end_date:
        args.append(f"--end-date {end_date}")
    if output_format:
        args.append(f"--output-format {output_format}")
    if show_metrics:
        args.append("--show-metrics")
    if forecast_horizon > 1:
        args.append(f"--forecast-horizon {forecast_horizon}")
    
    return " ".join(args)

def display_prediction_status():
    """Display prediction status section"""
    status_container = st.container()
    
    with status_container:
        st.markdown("##### 🔄 Prediction Status")
        st.markdown("---")
        progress_bar = st.empty()
        status_text = st.empty()
        return progress_bar, status_text

def get_available_models(models_dir: str) -> List[Dict]:
    """Get list of available models with detailed information"""
    available_models = []
    if os.path.exists(models_dir):
        for model_dir in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_dir)
            if os.path.isdir(model_path):
                try:
                    # Get model info
                    model_info = get_model_info(model_path)
                    
                    # Create model details
                    model_details = {
                        'Model Name': model_dir,
                        'Type': model_info.get('model_type', 'Unknown'),
                        'Training Date': model_info.get('training_date', 'Unknown'),
                        'Target': model_info.get('target', 'Unknown'),
                        'Features': ', '.join(model_info.get('features', []))[:50] + '...' if len(', '.join(model_info.get('features', []))) > 50 else ', '.join(model_info.get('features', []))
                    }
                    available_models.append(model_details)
                except Exception as e:
                    logging.error(f"Error getting details for model {model_dir}: {str(e)}")
                    continue
    return available_models

def on_ts_pred_model_selection_change():
    """Callback to handle model selection changes"""
    edited_rows = st.session_state['ts_pred_model_editor']['edited_rows']
    current_model = None
    
    # Get the model data from session state
    model_data = st.session_state['ts_pred_model_data']
    
    for idx, changes in edited_rows.items():
        if '🔍 Select' in changes:
            # Use the model data from session state
            model_name = model_data[idx]['Model Name']
            st.session_state['ts_pred_model_selections'][model_name] = changes['🔍 Select']
            if changes['🔍 Select']:
                current_model = model_name
    
    # Update selected model
    selected_models = [
        name for name, is_selected in st.session_state['ts_pred_model_selections'].items() 
        if is_selected
    ]
    
    if len(selected_models) > 1:
        # If multiple models are selected, keep only the most recent selection
        for model in selected_models[:-1]:
            st.session_state['ts_pred_model_selections'][model] = False
        current_model = selected_models[-1]
    
    st.session_state['ts_pred_selected_model'] = current_model

def on_ts_pred_table_selection_change():
    """Callback to handle table selection changes"""
    edited_rows = st.session_state['ts_pred_table_editor']['edited_rows']
    
    # Get the table data from session state
    table_data = st.session_state['ts_pred_table_data']
    
    for idx, changes in edited_rows.items():
        if '🔍 Select' in changes:
            table_name = table_data[idx]['Table Name']
            st.session_state['ts_pred_table_selections'][table_name] = changes['🔍 Select']
            if changes['🔍 Select']:
                st.session_state['ts_pred_selected_table'] = table_name
            elif st.session_state['ts_pred_selected_table'] == table_name:
                st.session_state['ts_pred_selected_table'] = None
    
    # Ensure only one table is selected
    selected_tables = [
        name for name, is_selected in st.session_state['ts_pred_table_selections'].items()
        if is_selected
    ]
    
    if len(selected_tables) > 1:
        # Keep only the most recent selection
        for table in selected_tables[:-1]:
            st.session_state['ts_pred_table_selections'][table] = False
        st.session_state['ts_pred_selected_table'] = selected_tables[-1]

def predict_time_series_page():
    """Streamlit interface for time series predictions"""
    # Initialize session state
    initialize_ts_pred_session_state()
    
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
        
        # Status Tab - Contains prediction status and progress
        with status_tab:
            # Model Evaluation Status (always at top)
            evaluation_progress_bar, evaluation_status = display_prediction_status()
            
            # Status message placeholder
            status_placeholder = st.empty()
            
            # Show stop message if exists
            if st.session_state.get('ts_pred_stop_message'):
                status_placeholder.warning(st.session_state['ts_pred_stop_message'])
            
            # Stop button placeholder
            stop_button_placeholder = st.empty()
            
            # Prediction output
            prediction_progress = st.empty()
        
        # Results Tab - Contains metrics and visualizations
        with results_tab:
            metrics_placeholder = st.empty()

    # Left Column - Configuration
    with left_col:
        st.markdown("""
            <p style='color: #666; margin: 0; font-size: 0.9em;'>Configure prediction parameters</p>
            <hr style='margin: 0.2em 0 0.7em 0;'>
        """, unsafe_allow_html=True)
        
        # Table Selection Section with enhanced information
        st.markdown("##### 📊 Select Table for Prediction")
        
        # Get available tables
        available_tables = get_available_tables(db_path)
        
        if not available_tables:
            st.warning("No strategy tables found in the database.")
            return
    
        # Create or use existing table data
        if not st.session_state['ts_pred_table_data']:
            table_data = []
            for t in available_tables:
                # Use the stored selection state or default to False
                is_selected = st.session_state['ts_pred_table_selections'].get(t['name'], False)
                table_data.append({
                    '🔍 Select': is_selected,
                    'Table Name': t['name'],
                    'Date Range': t['date_range'],
                    'Rows': t['total_rows'],
                    'Symbols': ', '.join(t['symbols'])
                })
            st.session_state['ts_pred_table_data'] = table_data
        
        table_df = pd.DataFrame(st.session_state['ts_pred_table_data'])
        
        # Display table information with checkboxes
        edited_df = st.data_editor(
            table_df,
            hide_index=True,
            key='ts_pred_table_editor',
            column_config={
                "🔍 Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select this table for prediction",
                    default=False,
                ),
                "Table Name": st.column_config.TextColumn(
                    "Table Name",
                    help="Name of the strategy table",
                    width="medium"
                ),
                "Date Range": st.column_config.TextColumn(
                    "Date Range",
                    help="Time range of the data",
                    width="medium"
                ),
                "Rows": st.column_config.NumberColumn(
                    "Rows",
                    help="Number of data points",
                    format="%d"
                ),
                "Symbols": st.column_config.TextColumn(
                    "Symbols",
                    help="Trading symbols in the table",
                    width="small"
                )
            },
            disabled=["Table Name", "Date Range", "Rows", "Symbols"],
            use_container_width=True,
            on_change=on_ts_pred_table_selection_change
        )
        
        # Show selected table details
        if st.session_state['ts_pred_selected_table']:
            selected_info = next((t for t in available_tables if t['name'] == st.session_state['ts_pred_selected_table']), None)
            if selected_info:
                with st.expander(f"📊 Selected Table Details", expanded=True):
                    st.write(f"**Date Range:** {selected_info['date_range']}")
                    st.write(f"**Total Rows:** {selected_info['total_rows']:,}")
                    st.write(f"**Symbols:** {', '.join(selected_info['symbols'])}")
        
        # Model Selection Section
        st.markdown("##### 🤖 Select Model")
        
        # Get available models with details
        available_models = get_available_models(models_dir)
        
        if not available_models:
            st.warning("No trained models found. Please train a model first.")
            return
        
        # Create or use existing model data
        if not st.session_state['ts_pred_model_data']:
            model_data = []
            for m in available_models:
                # Use the stored selection state or default to False
                is_selected = st.session_state['ts_pred_model_selections'].get(m['Model Name'], False)
                model_data.append({
                    '🔍 Select': is_selected,
                    'Model Name': m['Model Name'],
                    'Type': m['Type'],
                    'Training Date': m['Training Date'],
                    'Target': m['Target'],
                    'Features': m['Features']
                })
            st.session_state['ts_pred_model_data'] = model_data
        
        model_df = pd.DataFrame(st.session_state['ts_pred_model_data'])
        
        # Display model information with checkboxes
        edited_model_df = st.data_editor(
            model_df,
            hide_index=True,
            key='ts_pred_model_editor',
            column_config={
                "🔍 Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select this model for prediction",
                    default=False,
                ),
                "Model Name": st.column_config.TextColumn(
                    "Model Name",
                    help="Name of the trained model",
                    width="medium"
                ),
                "Type": st.column_config.TextColumn(
                    "Model Type",
                    help="Type of the model",
                    width="small"
                ),
                "Training Date": st.column_config.TextColumn(
                    "Training Date",
                    help="When the model was trained",
                    width="medium"
                ),
                "Target": st.column_config.TextColumn(
                    "Target",
                    help="Target column used for training",
                    width="small"
                ),
                "Features": st.column_config.TextColumn(
                    "Features",
                    help="Features used in training",
                    width="large"
                )
            },
            disabled=["Model Name", "Type", "Training Date", "Target", "Features"],
            use_container_width=True,
            on_change=on_ts_pred_model_selection_change
        )
        
        # Show selected model details
        if st.session_state['ts_pred_selected_model']:
            selected_model = next((m for m in available_models if m['Model Name'] == st.session_state['ts_pred_selected_model']), None)
            if selected_model:
                with st.expander(f"🤖 Selected Model Details", expanded=True):
                    st.write(f"**Model Type:** {selected_model['Type']}")
                    st.write(f"**Training Date:** {selected_model['Training Date']}")
                    st.write(f"**Target Column:** {selected_model['Target']}")
                    st.write("**Features:**")
                    st.write(selected_model['Features'])

            # Date range selection
            st.markdown("##### 📅 Date Range")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date (optional)", value=None)
            with col2:
                end_date = st.date_input("End Date (optional)", value=None)
            
            # Forecast Configuration
            st.markdown("##### 🔮 Forecast Configuration")
            forecast_horizon = st.number_input(
                "Number of Future Time Points to Predict",
                min_value=1,
                max_value=100,
                value=1,
                help="How many future time points to predict"
            )
            
            # Output configuration
            st.markdown("##### 💾 Output Configuration")
            output_format = st.selectbox(
                "Output Format",
                options=['csv', 'json'],
                help="Choose the format for saving predictions"
            )
            
            # Show metrics option
            show_metrics = st.checkbox("Show Model Metrics", value=True,
                                    help="Display model performance metrics")
            
            # Generate output path
            if st.session_state['ts_pred_selected_model']:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                default_output = f"predictions_{st.session_state['ts_pred_selected_model']}_{timestamp}.{output_format}"
                output_path = os.path.join('predictions', default_output)
                
                # Display equivalent command
                st.markdown("##### 💻 Equivalent Command")
                if st.session_state['ts_pred_selected_table']:
                    model_path = os.path.join(models_dir, st.session_state['ts_pred_selected_model'])
                    cmd = get_pred_equivalent_command(
                        model_path=model_path,
                        table_name=st.session_state['ts_pred_selected_table'],
                        output_path=output_path,
                        start_date=start_date.strftime("%Y-%m-%d") if start_date else None,
                        end_date=end_date.strftime("%Y-%m-%d") if end_date else None,
                        output_format=output_format,
                        show_metrics=show_metrics,
                        forecast_horizon=forecast_horizon
                    )
                    st.code(cmd, language='bash')
                
                # Run predictions button
                if st.button("🚀 Run Predictions", type="primary", use_container_width=True):
                    if not st.session_state['ts_pred_selected_table']:
                        st.error("⚠️ Please select a table for predictions.")
                    else:
                        try:
                            # Clear right side content and reset session state
                            st.session_state['ts_pred_stop_clicked'] = False
                            st.session_state['ts_pred_stop_message'] = None
                            
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
                                        on_click=on_ts_pred_stop_click,
                                        key="ts_pred_stop",
                                        help="Click to stop the prediction process",
                                        type="secondary")
                            
                            # Execute the command
                            with st.spinner("Running predictions..."):
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
                                    if check_ts_pred_stop_clicked():
                                        process.terminate()
                                        status_placeholder.warning("Prediction stopped by user")
                                        break
                                
                                # Wait for process to complete
                                process.wait()
                                
                                # Check return code
                                if process.returncode == 0:
                                    status_placeholder.success("✨ Predictions completed successfully")
                                    evaluation_progress_bar.progress(1.0)
                                    evaluation_status.text("Predictions completed successfully")
                                    
                                    predictions = None
                                    try:
                                        predictions_path = os.path.join(current_dir, output_path)
                                        if os.path.exists(predictions_path):
                                            if output_format == 'csv':
                                                predictions = pd.read_csv(predictions_path)
                                            else:  # json
                                                predictions = pd.read_json(predictions_path)
                                            
                                            with results_tab:
                                                st.markdown("### 📊 Prediction Results")
                                                st.dataframe(predictions)
                                    except Exception as e:
                                        st.error(f"Error loading predictions: {str(e)}")
                                    
                                    if predictions is not None:
                                        st.download_button(
                                            label="📥 Download Predictions",
                                            data=predictions.to_csv(index=False),
                                            file_name=default_output,
                                            mime=f"text/{output_format}"
                                        )
                                else:
                                    status_placeholder.error("❌ Predictions failed")
                                    evaluation_status.text("Predictions failed")
                        except Exception as e:
                            status_placeholder.error(f"Error during prediction: {str(e)}")
                            logging.error(f"Prediction error: {str(e)}", exc_info=True)
                        finally:
                            # Ensure stop button is removed in all cases
                            stop_button_placeholder.empty()

if __name__ == "__main__":
    predict_time_series_page() 