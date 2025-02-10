import streamlit as st
import pandas as pd
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional
from train_meta_model import train_meta_model
import json
import logging
from itertools import zip_longest

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
                source_table,
                MIN(id) as first_id,
                MAX(id) as last_id,
                COUNT(*) as prediction_count,
                MIN(datetime) as start_date,
                MAX(datetime) as end_date,
                AVG(predicted_price) as avg_prediction,
                AVG(actual_price) as avg_actual,
                AVG(ABS(predicted_price - actual_price)) as avg_error,
                MAX(id) as latest_id
            FROM historical_predictions
            GROUP BY run_id, model_name, source_table
            ORDER BY latest_id DESC
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return []
        
        # Convert to list of dictionaries with run information
        available_runs = []
        for row in results:
            run_id, model_name, source_table, first_id, last_id, count, start_date, end_date, avg_pred, avg_actual, avg_error, _ = row
            
            # Skip if run_id is None
            if not run_id:
                continue
            
            available_runs.append({
                'run_id': run_id,
                'model_name': model_name if model_name else 'Unknown',
                'source_table': source_table if source_table else 'Unknown',
                'id_range': {'first': first_id, 'last': last_id},
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
    try:
        # Display Meta-Model Name
        st.markdown(f"""
            <h4 style='color: #1565C0;'>Meta-Model Information</h4>
        """, unsafe_allow_html=True)
        
        model_name = meta_model_info.get('model_name', 'Unknown')
        if isinstance(model_name, dict):
            model_name = str(model_name)
        st.markdown(f"**Meta-Model Name:** `{model_name}`")
        
        # Create a table-like display for metrics
        st.markdown("""
            <h4 style='color: #1565C0;'>Performance Metrics</h4>
        """, unsafe_allow_html=True)
        
        # Filter and rename metrics for display
        display_metrics = {}
        if isinstance(metrics, dict):
            for key in ['train_rmse', 'test_rmse', 'train_mae', 'test_mae']:
                if key in metrics:
                    display_metrics[key] = metrics[key]
        
        # Create a DataFrame for the metrics
        if display_metrics:
            metrics_df = pd.DataFrame([display_metrics])
            st.dataframe(
                metrics_df,
                hide_index=True,
                use_container_width=True
            )
        else:
            st.write("No performance metrics available")
        
        # Training Information with better formatting
        st.markdown("""
            <h4 style='color: #1565C0; margin-top: 1em;'>Training Information</h4>
        """, unsafe_allow_html=True)
        
        # Get base model information safely
        base_model_run_ids = meta_model_info.get('base_model_run_ids', [])
        if isinstance(base_model_run_ids, str):
            base_model_run_ids = [base_model_run_ids]
        n_base_models = len(base_model_run_ids)
        
        # Create training info safely
        training_info = pd.DataFrame([{
            'Information': 'Number of Base Models',
            'Value': str(n_base_models)
        }])
        
        # Display training info
        st.dataframe(
            training_info,
            hide_index=True,
            use_container_width=True
        )
        
        # Display Base Models Information if available
        base_model_names = meta_model_info.get('base_model_names', [])
        if isinstance(base_model_names, str):
            base_model_names = [base_model_names]
            
        if base_model_names or base_model_run_ids:
            st.markdown("""
                <h4 style='color: #1565C0; margin-top: 1em;'>Base Models</h4>
            """, unsafe_allow_html=True)
            
            # Create a list of tuples of model names and run ids
            base_models = []
            if base_model_names and base_model_run_ids:
                base_models = list(zip_longest(base_model_names, base_model_run_ids))
            elif base_model_names:
                base_models = [(name, None) for name in base_model_names]
            elif base_model_run_ids:
                base_models = [(None, run_id) for run_id in base_model_run_ids]
            
            for i, (model_name, run_id) in enumerate(base_models, 1):
                model_info = []
                if model_name:
                    model_info.append(f"**Model:** `{model_name}`")
                if run_id:
                    model_info.append(f"**Run ID:** `{run_id}`")
                if model_info:
                    st.markdown(f"{i}. {' | '.join(model_info)}")
    
    except Exception as e:
        st.error(f"Error displaying metrics: {str(e)}")
        st.write("Raw Results:")
        st.json({"metrics": metrics, "meta_model_info": meta_model_info})

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
        
        # Initialize session state variables if they don't exist
        if 'meta_selected_runs' not in st.session_state:
            st.session_state['meta_selected_runs'] = []
        if 'meta_run_selections' not in st.session_state:
            st.session_state['meta_run_selections'] = {}
        
        def on_meta_run_selection_change():
            """Callback to handle run selection changes"""
            edited_rows = st.session_state['meta_model_run_editor']['edited_rows']
            for idx, changes in edited_rows.items():
                if '🔍 Select' in changes:
                    run_id = run_df.iloc[idx]['Run ID']
                    st.session_state['meta_run_selections'][run_id] = changes['🔍 Select']
            
            # Update selected runs list
            st.session_state['meta_selected_runs'] = [
                run_id for run_id, is_selected in st.session_state['meta_run_selections'].items() 
                if is_selected
            ]
        
        # Run Selection Section with enhanced information
        st.markdown("##### 🔄 Select Base Model Runs")
        
        # Create a DataFrame for better visualization of run information
        run_data = []
        for run in available_runs:
            # Use the stored selection state or default to False
            is_selected = st.session_state['meta_run_selections'].get(run['run_id'], False)
            # Format the period more compactly
            start_date = run['prediction_period']['start'].split()[0] if run['prediction_period']['start'] else ''
            end_date = run['prediction_period']['end'].split()[0] if run['prediction_period']['end'] else ''
            period = f"{start_date} to {end_date}" if start_date and end_date else "N/A"
            
            run_data.append({
                '🔍 Select': is_selected,  # Checkbox column
                'Run ID': run['run_id'],  # Show full run_id
                'Model': run['model_name'],  # Show full model name
                'Source': run['source_table'].replace('strategy_', ''),  # Keep removing strategy_ prefix
                'IDs': f"{run['id_range']['first']}-{run['id_range']['last']}",
                'Period': period,
                'Count': run['metrics']['prediction_count'],
                'Error': run['metrics']['avg_error']
            })
        
        run_df = pd.DataFrame(run_data)
        
        # Display run information with checkboxes
        st.markdown("##### Select runs from the list below:")
        edited_df = st.data_editor(
            run_df,
            hide_index=True,
            key="meta_model_run_editor",  # Unique key for the editor
            column_config={
                "🔍 Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select this run for meta-model training",
                    default=False,
                    width="small"
                ),
                "Run ID": st.column_config.TextColumn(
                    "Run ID",
                    help="Unique identifier for the prediction run",
                    width="medium"  # Increased width to accommodate full run_id
                ),
                "Model": st.column_config.TextColumn(
                    "Model",
                    help="Model used for predictions",
                    width="medium"  # Increased width to accommodate full model name
                ),
                "Source": st.column_config.TextColumn(
                    "Source",
                    help="Original strategy table used for predictions",
                    width="small"
                ),
                "IDs": st.column_config.TextColumn(
                    "IDs",
                    help="Range of prediction IDs in the database",
                    width="small"
                ),
                "Period": st.column_config.TextColumn(
                    "Period",
                    help="Time range of predictions",
                    width="medium"
                ),
                "Count": st.column_config.NumberColumn(
                    "Count",
                    help="Number of predictions in this run",
                    format="%d",
                    width="small"
                ),
                "Error": st.column_config.NumberColumn(
                    "Error",
                    help="Average prediction error",
                    format="%.4f",
                    width="small"
                )
            },
            disabled=["Run ID", "Model", "Source", "IDs", "Period", "Count", "Error"],
            use_container_width=True,
            on_change=on_meta_run_selection_change
        )
        
        # Update selected runs based on checkbox changes
        st.session_state.meta_selected_runs = [
            run_id for run_id, is_selected in st.session_state['meta_run_selections'].items() 
            if is_selected
        ]
        
        # Show selected runs details
        if st.session_state.meta_selected_runs:
            st.markdown("##### 📈 Selected Runs Details")
            selected_info = [run for run in available_runs if run['run_id'] in st.session_state.meta_selected_runs]
            
            # Create a summary of selected runs
            total_predictions = sum(run['metrics']['prediction_count'] for run in selected_info)
            unique_models = len(set(run['model_name'] for run in selected_info))
            
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Selected Runs", len(selected_info))
            with col2:
                st.metric("Total Predictions", f"{total_predictions:,}")
            with col3:
                st.metric("Unique Models", unique_models)
            
            # Show detailed information for each selected run
            for run in selected_info:
                with st.expander(f"📊 Run: {run['run_id']} ({run['model_name']})", expanded=True):
                    st.write(f"**Model:** {run['model_name']}")
                    st.write(f"**Source Table:** {run['source_table']}")
                    st.write(f"**ID Range:** {run['id_range']['first']} - {run['id_range']['last']}")
                    st.write(f"**Period:** {run['prediction_period']['start']} to {run['prediction_period']['end']}")
                    st.write(f"**Predictions:** {run['metrics']['prediction_count']:,}")
                    st.write(f"**Average Error:** {run['metrics']['avg_error']:.4f}")
                    if run['metrics']['avg_prediction'] and run['metrics']['avg_actual']:
                        st.write(f"**Avg Prediction:** {run['metrics']['avg_prediction']:.4f}")
                        st.write(f"**Avg Actual:** {run['metrics']['avg_actual']:.4f}")
        
        # Model Parameters Section
        st.markdown("##### 🤖 Meta-Model Parameters")
        model_params = get_meta_model_params()
        
        # Training Options
        st.markdown("##### ⚙️ Training Options")
        test_size = st.slider(
            "Test Size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.1,
            help="Proportion of data to use for testing"
        )
        
        force_retrain = st.checkbox(
            "Force Retrain",
            help="If checked, will perform a full retrain regardless of existing meta-models"
        )
        
        # Training Button
        st.markdown("##### 🚀 Start Training")
        train_button = st.button(
            "Train Meta-Model",
            type="primary",
            use_container_width=True,
            help="Click to start the meta-model training process",
            disabled=len(st.session_state.meta_selected_runs) < 2  # Disable if fewer than 2 runs selected
        )

    # Right Column - Output
    with right_col:
        st.markdown("""
            <p style='color: #666; margin: 0; font-size: 0.9em;'>Training output and results</p>
            <hr style='margin: 0.2em 0 0.7em 0;'>
        """, unsafe_allow_html=True)
        
        # Command Preview Section
        if len(st.session_state.meta_selected_runs) >= 2:
            st.markdown("##### 📝 Command Preview")
            cmd = get_equivalent_command(st.session_state.meta_selected_runs, test_size, force_retrain)
            st.code(cmd, language="bash")
        elif st.session_state.meta_selected_runs:
            st.warning("⚠️ Please select at least two runs for meta-model training")
        
        # Training Output Section
        if train_button:
            if len(st.session_state.meta_selected_runs) < 2:
                st.error("⚠️ Please select at least two runs for meta-model training.")
            else:
                try:
                    st.markdown("##### 🔄 Execution")
                    st.info("Starting meta-model training...")
                    
                    with st.spinner("Training meta-model... Please wait"):
                        results = train_meta_model(
                            run_ids=st.session_state.meta_selected_runs,
                            test_size=test_size,
                            force_retrain=force_retrain
                        )
                        
                        # Display metrics - handle different result structures
                        if isinstance(results, dict):
                            metrics = results.get('metrics', {})
                            meta_info = results.get('meta_model_info', results)  # fallback to entire results if no meta_model_info
                            if metrics or meta_info:
                                display_training_metrics(metrics, meta_info)
                            else:
                                st.json(results)  # Display raw results if no metrics found
                        else:
                            st.write("Training Results:", results)
                        
                    st.success("✅ Meta-model training completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")
                    logging.exception("Detailed error traceback:") 