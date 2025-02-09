import streamlit as st
import pandas as pd
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional
from meta_model_predictor import run_meta_predictions
import mlflow
import logging

def get_available_tables(db_path: str) -> List[str]:
    """Get list of available tables from the database, sorted by creation time (newest first)"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get tables from sqlite_master, sorted by name in descending order
        # Strategy tables are named with timestamp, so DESC order will show newest first
        cursor.execute("""
            SELECT name 
            FROM sqlite_master 
            WHERE type='table' 
            AND name LIKE 'strategy_%'
            ORDER BY name DESC;
        """)
        
        tables = [table[0] for table in cursor.fetchall()]
        conn.close()
        
        if tables:
            logging.info(f"Found {len(tables)} strategy tables")
            logging.info(f"Most recent tables: {tables[:3]}")
        
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
        
        # Table Selection Section
        st.markdown("##### 📊 Select Table for Prediction")
        selected_table = st.selectbox(
            label="Select Table",
            options=available_tables,
            key="meta_model_predictor_selected_table",
            label_visibility="collapsed",
            help="Choose a table to run predictions on"
        )
        
        # Model Selection Section
        st.markdown("##### 🤖 Select Meta Model")
        selected_model = st.selectbox(
            label="Select Meta Model",
            options=available_models,
            key="meta_model_predictor_selected_model",
            label_visibility="collapsed",
            help="Choose a meta model to use for predictions"
        )
        
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
        if selected_table and selected_model:
            st.markdown("##### 📝 Command Preview")
            cmd = get_equivalent_command(selected_table, selected_model, force_new_run)
            st.code(cmd, language="bash")
        
        # Prediction Output Section
        if predict_button:
            try:
                st.markdown("##### 🔄 Execution")
                st.info("Executing command:")
                st.code(get_equivalent_command(selected_table, selected_model, force_new_run), language="bash")
                
                with st.spinner("🔄 Running predictions... Please wait"):
                    results = run_meta_predictions(
                        table_name=selected_table,
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