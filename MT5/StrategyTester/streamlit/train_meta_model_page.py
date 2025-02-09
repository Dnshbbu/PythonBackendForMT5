import streamlit as st
import pandas as pd
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional
from train_meta_model import train_meta_model
import json
import logging

def get_available_runs(db_path: str) -> List[Dict]:
    """Get list of available runs from historical_predictions table"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get run_ids and their statistics from historical_predictions
        cursor.execute("""
            SELECT 
                run_id,
                model_name,
                COUNT(*) as prediction_count,
                MIN(datetime) as start_date,
                MAX(datetime) as end_date,
                AVG(predicted_price) as avg_prediction,
                AVG(actual_price) as avg_actual,
                AVG(ABS(predicted_price - actual_price)) as avg_error,
                MAX(id) as latest_id
            FROM historical_predictions
            GROUP BY run_id, model_name
            ORDER BY latest_id DESC
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return []
        
        # Convert to list of dictionaries with run information
        available_runs = []
        for row in results:
            run_id, model_name, count, start_date, end_date, avg_pred, avg_actual, avg_error, _ = row
            
            # Skip if run_id is None
            if not run_id:
                continue
            
            available_runs.append({
                'run_id': run_id,
                'model_name': model_name if model_name else 'Unknown',
                'metrics': {
                    'prediction_count': count,
                    'avg_prediction': round(avg_pred, 4) if avg_pred else None,
                    'avg_actual': round(avg_actual, 4) if avg_actual else None,
                    'avg_error': round(avg_error, 4) if avg_error else None
                },
                'prediction_period': {
                    'start': start_date,
                    'end': end_date
                }
            })
        
        return available_runs
            
    except Exception as e:
        st.error(f"Error accessing database: {str(e)}")
        st.exception(e)  # This will show the full traceback
        return []

def get_meta_model_params() -> Dict:
    """Get meta-model parameters"""
    return {
        'max_depth': st.slider('Max Depth', 3, 10, 6, key="meta_max_depth"),
        'learning_rate': st.number_input('Learning Rate', 0.01, 0.5, 0.05, 0.01, key="meta_learning_rate"),
        'n_estimators': st.number_input('Number of Estimators', 100, 2000, 1000, 100, key="meta_n_estimators"),
        'subsample': st.slider('Subsample', 0.5, 1.0, 0.8, 0.1, key="meta_subsample"),
        'colsample_bytree': st.slider('Column Sample by Tree', 0.5, 1.0, 0.8, 0.1, key="meta_colsample_bytree"),
        'min_child_weight': st.number_input('Min Child Weight', 1, 10, 2, 1, key="meta_min_child_weight")
    }

def display_training_metrics(metrics: Dict, meta_model_info: Dict):
    """Display training metrics in a formatted way"""
    if not metrics:
        return
    
    # Display Meta-Model Name
    st.markdown(f"""
        <h4 style='color: #1565C0;'>Meta-Model Information</h4>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**Meta-Model Name:** `{meta_model_info.get('model_name', 'Unknown')}`")
    
    # Create a table-like display for metrics
    st.markdown("""
        <h4 style='color: #1565C0;'>Performance Metrics</h4>
    """, unsafe_allow_html=True)
    
    # Filter and rename metrics for display
    display_metrics = {
        'train_rmse': metrics.get('train_rmse', None),
        'test_rmse': metrics.get('test_rmse', None),
        'train_mae': metrics.get('train_mae', None),
        'test_mae': metrics.get('test_mae', None)
    }
    
    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame([display_metrics])
    st.dataframe(
        metrics_df,
        hide_index=True,
        use_container_width=True
    )
    
    # Training Information with better formatting
    st.markdown("""
        <h4 style='color: #1565C0; margin-top: 1em;'>Training Information</h4>
    """, unsafe_allow_html=True)
    
    # Get base model count from meta_model_info
    n_base_models = len(meta_model_info.get('base_model_run_ids', [])) if 'base_model_run_ids' in meta_model_info else 0
    
    # Get database path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
    
    # Get training information from database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get total prediction count and date range for the selected run_ids
        cursor.execute("""
            SELECT 
                COUNT(*) as total_predictions,
                MIN(datetime) as start_date,
                MAX(datetime) as end_date
            FROM historical_predictions
            WHERE run_id IN ({})
        """.format(','.join(['?' for _ in meta_model_info['base_model_run_ids']])), 
        meta_model_info['base_model_run_ids'])
        
        total_predictions, start_date, end_date = cursor.fetchone()
        conn.close()
        
        training_period = f"{start_date} to {end_date}" if start_date and end_date else "Not available"
        
        # Create training info DataFrame
        training_info = pd.DataFrame([{
            'Information': info,
            'Value': value
        } for info, value in [
            ('Number of Base Models', str(n_base_models)),
            ('Training Samples', str(total_predictions)),
            ('Training Period', training_period)
        ]])
        
    except Exception as e:
        st.error(f"Error getting training information: {str(e)}")
        training_info = pd.DataFrame([{
            'Information': info,
            'Value': value
        } for info, value in [
            ('Number of Base Models', str(n_base_models)),
            ('Training Samples', 'Not available'),
            ('Training Period', 'Not available')
        ]])
    
    # Display training info in a consistent format
    st.dataframe(
        training_info,
        hide_index=True,
        use_container_width=True
    )
    
    # Display Base Models Information
    if 'base_model_names' in meta_model_info and 'base_model_run_ids' in meta_model_info:
        st.markdown("""
            <h4 style='color: #1565C0; margin-top: 1em;'>Base Models</h4>
        """, unsafe_allow_html=True)
        
        for i, (model_name, run_id) in enumerate(zip(
            meta_model_info.get('base_model_names', []), 
            meta_model_info.get('base_model_run_ids', [])
        )):
            st.markdown(f"{i+1}. **Model:** `{model_name}`")
            st.markdown(f"   **Run ID:** `{run_id}`")

def display_run_info(run: Dict):
    """Display run information in a formatted way"""
    st.markdown(f"**Run ID:** `{run['run_id']}`")
    
    # Display model name
    model_name = run.get('model_name', 'Unknown')
    st.markdown(f"**Model Name:** `{model_name}`")
    
    # Display metrics in a clean format
    if run['metrics']:
        metrics_df = pd.DataFrame([run['metrics']])
        st.dataframe(metrics_df, use_container_width=True)
    
    # Display prediction period
    if 'prediction_period' in run:
        st.markdown(f"**Prediction Period:** {run['prediction_period']['start']} to {run['prediction_period']['end']}")

def get_equivalent_command(run_ids: List[str], test_size: float, force_retrain: bool) -> str:
    """Generate the equivalent command line command"""
    base_cmd = "python train_meta_model.py"
    run_ids_arg = f"--run-ids {' '.join(run_ids)}"
    test_size_arg = f"--test-size {test_size}"
    force_arg = "--force-retrain" if force_retrain else ""
    
    return f"{base_cmd} {run_ids_arg} {test_size_arg} {force_arg}".strip()

def train_meta_model_page():
    """Streamlit interface for meta-model training"""
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
    
    # Get available runs
    available_runs = get_available_runs(db_path)
    if not available_runs:
        st.warning("No model runs with predictions found in the database.")
        return

    # Create main left-right layout
    left_col, right_col = st.columns([1, 1], gap="large")

    # Left Column - Configuration
    with left_col:
        st.markdown("""
            <p style='color: #666; margin: 0; font-size: 0.9em;'>Configure your meta-model training parameters</p>
            <hr style='margin: 0.2em 0 0.7em 0;'>
        """, unsafe_allow_html=True)
        
        # Run Selection Section
        st.markdown("##### 🔄 Select Base Model Runs")
        selected_runs = st.multiselect(
            label="Select Base Model Runs",
            options=[run['run_id'] for run in available_runs],
            key="meta_selected_runs",
            help="Choose two or more model runs to combine into a meta-model"
        )
        
        # Display selected run information
        if selected_runs:
            st.markdown("##### 📊 Selected Runs Information")
            for run_id in selected_runs:
                run_info = next((run for run in available_runs if run['run_id'] == run_id), None)
                if run_info:
                    with st.expander(f"Run: {run_id}", expanded=False):
                        display_run_info(run_info)
        
        # Parameters Section
        with st.expander("##### ⚙️ Advanced Parameters", expanded=False):
            test_size = st.slider(
                "Test Size",
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05,
                help="Proportion of data to use for testing",
                key="meta_test_size"
            )
            
            st.markdown("##### Meta-Model Parameters")
            meta_model_params = get_meta_model_params()
        
        # Options Section
        st.markdown("##### 🛠️ Options")
        force_retrain = st.checkbox(
            "Force Retrain",
            help="If checked, will perform a full retrain regardless of existing models",
            key="meta_force_retrain"
        )
        
        # Training Button
        st.markdown("##### 🚀 Start Training")
        train_button = st.button(
            "Train Meta-Model",
            type="primary",
            use_container_width=True,
            help="Click to start the meta-model training process",
            key="meta_train_button"
        )

    # Right Column - Output
    with right_col:
        st.markdown("""
            <p style='color: #666; margin: 0; font-size: 0.9em;'>Training output and results</p>
            <hr style='margin: 0.2em 0 0.7em 0;'>
        """, unsafe_allow_html=True)
        
        # Command Preview Section
        if selected_runs:
            st.markdown("##### 📝 Command Preview")
            cmd = get_equivalent_command(selected_runs, test_size, force_retrain)
            st.code(cmd, language="bash")
            
            # Validation messages
            if len(selected_runs) < 2:
                st.warning("⚠️ Please select at least two runs for meta-model training")
        
        # Training Output Section
        if train_button:
            if len(selected_runs) < 2:
                st.error("⚠️ Please select at least two runs for meta-model training.")
            else:
                try:
                    with st.spinner("🔄 Training meta-model... Please wait"):
                        meta_model_info = train_meta_model(
                            run_ids=selected_runs,
                            test_size=test_size,
                            force_retrain=force_retrain
                        )
                        
                        st.success("✅ Meta-model training completed successfully!")
                        
                        # Display results in a formatted way
                        st.markdown("##### 📊 Training Results")
                        
                        # Create tabs for different result views
                        result_tabs = st.tabs(["Summary", "Detailed Results"])
                        
                        with result_tabs[0]:
                            # Display key metrics
                            if 'metrics' in meta_model_info:
                                display_training_metrics(meta_model_info['metrics'], meta_model_info)
                            
                        with result_tabs[1]:
                            # Show full results in an expander
                            with st.expander("Full Results", expanded=False):
                                st.json(meta_model_info)
                            
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")
                    st.exception(e)  # This will show the full traceback 