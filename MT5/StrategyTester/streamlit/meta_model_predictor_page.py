import streamlit as st
import pandas as pd
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional
from meta_model_predictor import run_meta_predictions
import mlflow
import logging

def get_available_tables(db_path: str) -> List[Dict]:
    """Get list of available tables from the database with detailed information
    
    Returns:
        List of dictionaries containing table information:
        {
            'name': str,            # Table name
            'date_range': str,      # Date range of data
            'total_rows': int,      # Number of rows
            'symbols': List[str],   # List of symbols in the table
            'display_name': str     # Formatted name for display
        }
    """
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
        
        if tables:
            logging.info(f"Found {len(tables)} strategy tables")
            for table in tables[:3]:  # Log details of first 3 tables
                logging.info(f"Table: {table['name']}")
                logging.info(f"  Date Range: {table['date_range']}")
                logging.info(f"  Rows: {table['total_rows']}")
                logging.info(f"  Symbols: {', '.join(table['symbols'])}")
        
        return tables
    
    except Exception as e:
        st.error(f"Error accessing database: {str(e)}")
        return []

def get_available_meta_models(db_path: str) -> List[str]:
    """Get list of available meta models from the database, sorted by creation time (newest first)"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get meta models from the database, sorted by name in descending order
        # Meta model names include timestamps, so DESC order will show newest first
        cursor.execute("""
            SELECT model_name 
            FROM model_repository 
            WHERE model_name LIKE 'meta_%'
            ORDER BY model_name DESC;
        """)
        
        models = [model[0] for model in cursor.fetchall()]
        conn.close()
        
        if models:
            logging.info(f"Found {len(models)} meta models")
            logging.info(f"Most recent models: {models[:3]}")
        
        return models
    except Exception as e:
        st.error(f"Error accessing database: {str(e)}")
        return []

def get_base_models_for_meta_model(db_path: str, meta_model_name: str) -> List[str]:
    """Get list of base models associated with a meta model"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get base models from the model repository
        cursor.execute("""
            SELECT additional_metadata
            FROM model_repository 
            WHERE model_name = ?
            ORDER BY created_at DESC
            LIMIT 1;
        """, (meta_model_name,))  # Pass the parameter as a tuple
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            try:
                metadata = eval(result[0])  # Convert string representation to dict
                base_models = metadata.get('base_model_names', [])
                logging.info(f"Found base models for {meta_model_name}: {base_models}")
                return base_models
            except Exception as e:
                logging.error(f"Error parsing metadata for {meta_model_name}: {e}")
                return []
        
        logging.warning(f"No metadata found for meta model: {meta_model_name}")
        return []
    except Exception as e:
        st.error(f"Error getting base models: {str(e)}")
        logging.exception(f"Detailed error when getting base models for {meta_model_name}:")
        return []

def display_prediction_metrics(metrics: Dict):
    """Display prediction metrics in a formatted way"""
    if not metrics:
        return
    
    # Create a table-like display for metrics
    st.markdown("""
        <h4 style='color: #1565C0;'>Performance Metrics</h4>
    """, unsafe_allow_html=True)
    
    # Create a DataFrame for the metrics
    metrics_data = []
    metric_columns = []
    
    # Add metrics with better formatting
    metrics_mapping = {
        'mae': ('Mean Absolute Error', '📉'),
        'rmse': ('Root Mean Squared Error', '📊'),
        'mape': ('Mean Absolute Percentage Error', '📈'),
        'direction_accuracy': ('Direction Accuracy', '🎯')
    }
    
    for key, (display_name, emoji) in metrics_mapping.items():
        if key in metrics:
            value = metrics[key]
            if key == 'direction_accuracy':
                value = f"{value:.2f}%"
            elif key == 'mape':
                value = f"{value:.2f}%"
            else:
                value = f"{value:.4f}"
            metrics_data.append(value)
            metric_columns.append(f"{emoji} {display_name}")
    
    # Create DataFrame and display with styling
    if metrics_data:
        metrics_df = pd.DataFrame([metrics_data], columns=metric_columns)
        st.dataframe(
            metrics_df,
            hide_index=True,
            use_container_width=True
        )

def get_equivalent_command(table_name: str, model_name: Optional[str], force_new_run: bool) -> str:
    """Generate the equivalent command line command"""
    base_cmd = "python meta_model_predictor.py"
    table_arg = f"--table {table_name}"
    model_arg = f"--model {model_name}" if model_name else ""
    force_arg = "--force" if force_new_run else ""
    
    return f"{base_cmd} {table_arg} {model_arg} {force_arg}".strip()

def meta_model_predictor_page():
    """Streamlit interface for meta model predictions"""
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
    
    # Get available tables and models
    available_tables = get_available_tables(db_path)
    available_models = get_available_meta_models(db_path)
    
    if not available_tables:
        st.warning("No strategy tables found in the database.")
        return
    
    if not available_models:
        st.warning("No meta models found in the database.")
        return

    # Create main left-right layout
    left_col, right_col = st.columns([1, 1], gap="large")

    # Left Column - Configuration
    with left_col:
        st.markdown("""
            <p style='color: #666; margin: 0; font-size: 0.9em;'>Configure your meta model prediction parameters</p>
            <hr style='margin: 0.2em 0 0.7em 0;'>
        """, unsafe_allow_html=True)
        
        # Table Selection Section with enhanced information
        st.markdown("##### 📊 Select Table for Prediction")
        
        # Initialize session state for selected table if not exists
        if 'meta_predictor_selected_table' not in st.session_state:
            st.session_state.meta_predictor_selected_table = None
        
        # Create a DataFrame for better visualization of table information
        table_data = []
        for t in available_tables:
            # Check if this table is selected
            is_selected = t['name'] == st.session_state.meta_predictor_selected_table
            table_data.append({
                '🔍 Select': is_selected,  # Checkbox column
                'Table Name': t['name'],
                'Date Range': t['date_range'],
                'Rows': t['total_rows'],
                'Symbols': ', '.join(t['symbols'])
            })
        
        table_df = pd.DataFrame(table_data)
        
        # Display table information with checkboxes
        st.markdown("##### Select a table from the list below:")
        edited_df = st.data_editor(
            table_df,
            hide_index=True,
            key="meta_model_predictor_table_editor",
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
            use_container_width=True
        )
        
        # Update selected table based on checkbox changes
        selected_indices = edited_df[edited_df['🔍 Select']].index
        if len(selected_indices) > 0:
            # Take the first selected table (since we only want one)
            st.session_state.meta_predictor_selected_table = edited_df.loc[selected_indices[0], 'Table Name']
            if len(selected_indices) > 1:
                st.warning("⚠️ Only one table can be selected for prediction. Using the first selected table.")
        else:
            st.session_state.meta_predictor_selected_table = None
        
        # Show selected table details
        selected_info = None  # Initialize selected_info
        if st.session_state.meta_predictor_selected_table:
            selected_info = next((t for t in available_tables if t['name'] == st.session_state.meta_predictor_selected_table), None)
            if selected_info:
                with st.expander(f"📊 Selected Table Details", expanded=True):
                    st.write(f"**Date Range:** {selected_info['date_range']}")
                    st.write(f"**Total Rows:** {selected_info['total_rows']:,}")
                    st.write(f"**Symbols:** {', '.join(selected_info['symbols'])}")
        
        # Model Selection Section
        st.markdown("##### 🤖 Select Meta Model")
        selected_model = st.selectbox(
            label="Select Meta Model",
            options=available_models,
            key="meta_model_predictor_selected_model",
            label_visibility="collapsed",
            help="Choose a meta model to use for predictions"
        )
        
        # Display base models if a meta model is selected
        if selected_model:
            base_models = get_base_models_for_meta_model(db_path, selected_model)
            if base_models:
                st.markdown("##### 🔗 Base Models")
                for i, model in enumerate(base_models, 1):
                    st.markdown(f"""
                        <div style='
                            padding: 8px 12px;
                            border-radius: 4px;
                            background-color: #2E303D;
                            border: 1px solid #3E4049;
                            margin: 4px 0;
                            font-size: 0.9em;
                            color: #E0E0E0;
                        '>
                            {i}. {model}
                        </div>
                    """, unsafe_allow_html=True)
        
        # Options Section
        st.markdown("##### ⚙️ Options")
        force_new_run = st.checkbox(
            "Force New Run",
            key="meta_model_predictor_force_new_run",
            help="If checked, will perform a new prediction run regardless of existing predictions"
        )
        
        # Prediction Button
        st.markdown("##### 🚀 Start Prediction")
        predict_button = st.button(
            "Run Predictions",
            key="meta_model_predictor_run_button",
            type="primary",
            use_container_width=True,
            help="Click to start the prediction process"
        )

    # Right Column - Output
    with right_col:
        st.markdown("""
            <p style='color: #666; margin: 0; font-size: 0.9em;'>Prediction output and results</p>
            <hr style='margin: 0.2em 0 0.7em 0;'>
        """, unsafe_allow_html=True)
        
        # Command Preview Section
        if selected_model and selected_info:  # Check both conditions
            st.markdown("##### 📝 Command Preview")
            cmd = get_equivalent_command(selected_info['name'], selected_model, force_new_run)
            st.code(cmd, language="bash")
        elif selected_model:  # Show message if table is not selected
            st.warning("Please select a table to preview the command")
        
        # Prediction Output Section
        if predict_button:
            if not selected_info:
                st.error("Please select a table before running predictions")
                return
                
            try:
                st.markdown("##### 🔄 Execution")
                st.info("Executing command:")
                st.code(get_equivalent_command(selected_info['name'], selected_model, force_new_run), language="bash")
                
                with st.spinner("🔄 Running predictions... Please wait"):
                    results = run_meta_predictions(
                        table_name=selected_info['name'],
                        meta_model_name=selected_model,
                        force_new_run=force_new_run
                    )
                    
                    # Display metrics
                    display_prediction_metrics(results['metrics'])
                    
                st.success("✅ Predictions completed successfully and stored in historical_predictions table!")
                
            except Exception as e:
                st.error(f"❌ Error running predictions: {str(e)}")
                logging.exception("Detailed error traceback:")

if __name__ == "__main__":
    meta_model_predictor_page() 