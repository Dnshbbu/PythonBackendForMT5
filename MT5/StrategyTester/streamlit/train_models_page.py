import streamlit as st
import pandas as pd
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional
from train_models import train_single_table, train_multi_table, train_model_incrementally
import mlflow
import torch
import json
import logging

def get_available_tables(db_path: str) -> List[str]:
    """Get list of available tables from the database, sorted by creation time (newest first)"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get tables with their creation time from sqlite_master
        cursor.execute("""
            SELECT name, sql, tbl_name
            FROM sqlite_master 
            WHERE type='table' AND name LIKE 'strategy_%'
            ORDER BY name DESC;  -- This will sort by name in descending order, 
                               -- assuming your table names include timestamps
        """)
        
        tables = [table[0] for table in cursor.fetchall()]
        conn.close()
        
        # If you want to see the table names in the logs
        if tables:
            logging.info(f"Found {len(tables)} strategy tables")
            logging.info(f"Most recent tables: {tables[:3]}")
        
        return tables
    except Exception as e:
        st.error(f"Error accessing database: {str(e)}")
        return []

def get_model_types() -> List[str]:
    """Get available model types"""
    return ['xgboost', 'random_forest', 'decision_tree', 'lstm']

def get_model_params(model_type: str) -> Dict:
    """Get default model parameters based on model type"""
    if model_type == 'xgboost':
        return {
            'max_depth': st.slider('Max Depth', 3, 10, 6),
            'learning_rate': st.number_input('Learning Rate', 0.01, 0.5, 0.05, 0.01),
            'n_estimators': st.number_input('Number of Estimators', 100, 2000, 1000, 100),
            'subsample': st.slider('Subsample', 0.5, 1.0, 0.8, 0.1),
            'colsample_bytree': st.slider('Column Sample by Tree', 0.5, 1.0, 0.8, 0.1),
            'min_child_weight': st.number_input('Min Child Weight', 1, 10, 2, 1)
        }
    elif model_type == 'lstm':
        return {
            'hidden_size': st.number_input('Hidden Size', 16, 256, 64, 16),
            'num_layers': st.number_input('Number of Layers', 1, 5, 2, 1),
            'sequence_length': st.number_input('Sequence Length', 5, 50, 10, 5),
            'batch_size': st.number_input('Batch Size', 16, 128, 32, 16),
            'learning_rate': st.number_input('Learning Rate', 0.0001, 0.01, 0.001, 0.0001),
            'num_epochs': st.number_input('Number of Epochs', 5, 50, 10, 5),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    elif model_type == 'random_forest':
        return {
            'n_estimators': st.number_input('Number of Estimators', 50, 500, 100, 50),
            'max_depth': st.slider('Max Depth', 3, 20, 10, 1),
            'min_samples_split': st.number_input('Min Samples Split', 2, 10, 2, 1),
            'min_samples_leaf': st.number_input('Min Samples Leaf', 1, 10, 1, 1)
        }
    else:  # decision_tree
        return {
            'max_depth': st.slider('Max Depth', 3, 20, 8, 1),
            'min_samples_split': st.number_input('Min Samples Split', 2, 10, 5, 1),
            'min_samples_leaf': st.number_input('Min Samples Leaf', 1, 10, 2, 1)
        }

def display_training_metrics(metrics: Dict):
    """Display training metrics in a formatted way"""
    if not metrics:
        return
    
    # Create a table-like display for metrics
    st.markdown("""
        <h4 style='color: #1565C0;'>Performance Metrics</h4>
    """, unsafe_allow_html=True)
    
    # Create a DataFrame for the metrics
    metrics_data = []
    metric_columns = []
    
    # Add basic metrics with better formatting
    metrics_mapping = {
        'training_loss': ('Training Loss', '📉'),
        'rmse': ('RMSE', '📊'),
        'n_features': ('Number of Features', '🔢'),
        'training_samples': ('Training Samples', '📈'),
        'sequence_length': ('Sequence Length', '📏'),
        'hidden_size': ('Hidden Size', '🔍'),
        'num_layers': ('Number of Layers', '📚'),
        'data_points': ('Data Points', '📊')
    }
    
    for key, (display_name, emoji) in metrics_mapping.items():
        if key in metrics:
            metrics_data.append(metrics[key])
            metric_columns.append(f"{emoji} {display_name}")
    
    # Create DataFrame and display with styling
    if metrics_data:
        metrics_df = pd.DataFrame([metrics_data], columns=metric_columns)
        st.dataframe(
            metrics_df,
            hide_index=True,
            use_container_width=True
        )
    
    # Training Information with better formatting
    st.markdown("""
        <h4 style='color: #1565C0; margin-top: 1em;'>Training Information</h4>
    """, unsafe_allow_html=True)
    
    training_info = {
        "📊 Number of Features": metrics.get('n_features', ''),
        "📈 Number of Samples": metrics.get('training_samples', ''),
        "📅 Training Period": f"{metrics.get('training_period', {}).get('start', '')} to {metrics.get('training_period', {}).get('end', '')}"
    }
    st.json(training_info)

def get_equivalent_command(training_mode: str, selected_tables: List[str], model_types: List[str], force_retrain: bool) -> str:
    """Generate the equivalent command line command"""
    base_cmd = "python train_models.py"
    mode_arg = f"--mode {training_mode}"
    tables_arg = f"--tables {' '.join(selected_tables)}"
    model_types_arg = f"--model-types {' '.join(model_types)}"
    force_arg = "--force-retrain" if force_retrain else ""
    
    return f"{base_cmd} {mode_arg} {tables_arg} {model_types_arg} {force_arg}".strip()

def train_models_page():
    """Streamlit interface matching the command line interface of train_models.py"""
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
    
    # Get available tables
    available_tables = get_available_tables(db_path)
    if not available_tables:
        st.warning("No strategy tables found in the database.")
        return

    # Create main left-right layout
    left_col, right_col = st.columns([1, 1], gap="large")

    # Left Column - Configuration
    with left_col:
        st.markdown("""
            <p style='color: #666; margin: 0; font-size: 0.9em;'>Configure your model training parameters</p>
            <hr style='margin: 0.2em 0 0.7em 0;'>
        """, unsafe_allow_html=True)
        
        # Table Selection Section
        st.markdown("##### 📊 Select Tables for Training")
        selected_tables = st.multiselect(
            label="Select Tables",
            options=available_tables,
            key="selected_tables",
            label_visibility="collapsed",
            help="Choose one or more tables for training your model"
        )
        
        # Model Selection Section
        st.markdown("##### 🤖 Select Model Types")
        model_types = st.multiselect(
            label="Select Model Types",
            options=['xgboost', 'decision_tree', 'random_forest', 'lstm'],
            default=['lstm'],
            key="model_types",
            label_visibility="collapsed",
            help="Choose one or more model types to train"
        )
        
        # Training Type Section
        st.markdown("##### 🔄 Training Type")
        training_mode = st.radio(
            label="Training Mode",
            options=["single", "multi", "incremental"],
            horizontal=True,  # Changed to horizontal to save space
            key="training_mode",
            label_visibility="collapsed",
            help="Choose the type of training to perform"
        )
        
        # Options Section
        st.markdown("##### ⚙️ Options")
        force_retrain = st.checkbox(
            "Force Retrain",
            help="If checked, will perform a full retrain regardless of existing models"
        )
        
        # Training Button
        st.markdown("##### 🚀 Start Training")
        train_button = st.button(
            "Train Models",
            type="primary",
            use_container_width=True,
            help="Click to start the training process"
        )

    # Right Column - Output
    with right_col:
        st.markdown("""
            <p style='color: #666; margin: 0; font-size: 0.9em;'>Training output and results</p>
            <hr style='margin: 0.2em 0 0.7em 0;'>
        """, unsafe_allow_html=True)
        
        # Command Preview Section
        if selected_tables and model_types:
            st.markdown("##### 📝 Command Preview")
            cmd = get_equivalent_command(training_mode, selected_tables, model_types, force_retrain)
            st.code(cmd, language="bash")
            
            # Add validation messages
            if training_mode == "single" and len(selected_tables) > 1:
                st.warning("⚠️ Single mode requires exactly one table")
            elif (training_mode in ["multi", "incremental"]) and len(selected_tables) < 2:
                st.warning("⚠️ This mode requires at least two tables")
        
        # Training Output Section
        if train_button:
            if not selected_tables:
                st.error("⚠️ Please select at least one table for training.")
            elif not model_types:
                st.error("⚠️ Please select at least one model type.")
            else:
                try:
                    st.markdown("##### 🔄 Execution")
                    st.info("Executing command:")
                    st.code(get_equivalent_command(training_mode, selected_tables, model_types, force_retrain), language="bash")
                    
                    with st.spinner("🔄 Training models... Please wait"):
                        if training_mode == "single":
                            if len(selected_tables) != 1:
                                st.error("⚠️ Single mode requires exactly one table")
                                return
                            results = train_single_table(
                                table_name=selected_tables[0],
                                force_retrain=force_retrain,
                                model_types=model_types
                            )
                        elif training_mode == "multi":
                            if len(selected_tables) < 2:
                                st.error("⚠️ Multi-table training requires at least two tables.")
                                return
                            results = train_multi_table(
                                table_names=selected_tables,
                                force_retrain=force_retrain,
                                model_types=model_types
                            )
                        else:  # incremental
                            if len(selected_tables) < 2:
                                st.error("⚠️ Incremental mode requires at least two tables")
                                return
                            base_table = selected_tables[0]
                            new_tables = selected_tables[1:]
                            results = train_model_incrementally(
                                base_table=base_table,
                                new_tables=new_tables,
                                force_retrain=force_retrain,
                                model_types=model_types
                            )
                        
                        st.success("✅ Training completed successfully!")
                        
                        # Display results in a formatted way
                        st.markdown("##### 📊 Training Results")
                        
                        # Create tabs for different result views
                        result_tabs = st.tabs(["Summary", "Detailed Results"])
                        
                        with result_tabs[0]:
                            if isinstance(results, dict):
                                # Display key metrics in a more condensed format
                                metrics_to_show = {
                                    k: v for k, v in results.items() 
                                    if isinstance(v, (int, float, str)) or 
                                    (isinstance(v, dict) and 'metrics' in v)
                                }
                                st.json(metrics_to_show)
                        
                        with result_tabs[1]:
                            # Show full results in an expander
                            with st.expander("Full Results", expanded=False):
                                st.json(results)
                            
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}") 