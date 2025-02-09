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

def get_available_tables(db_path: str) -> List[str]:
    """Get list of available tables from the database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        conn.close()
        return [table for table in tables if table.startswith('strategy_')]
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
    st.write("Performance Metrics:")
    
    # Create a DataFrame for the metrics
    metrics_data = []
    metric_columns = []
    
    # Add basic metrics
    if 'training_loss' in metrics:
        metrics_data.append(metrics['training_loss'])
        metric_columns.append('training_loss')
    if 'rmse' in metrics:
        metrics_data.append(metrics['rmse'])
        metric_columns.append('rmse')
    if 'n_features' in metrics:
        metrics_data.append(metrics['n_features'])
        metric_columns.append('n_features')
    if 'training_samples' in metrics:
        metrics_data.append(metrics['training_samples'])
        metric_columns.append('n_samples')
    if 'sequence_length' in metrics:
        metrics_data.append(metrics['sequence_length'])
        metric_columns.append('sequence_length')
    if 'hidden_size' in metrics:
        metrics_data.append(metrics['hidden_size'])
        metric_columns.append('hidden_size')
    if 'num_layers' in metrics:
        metrics_data.append(metrics['num_layers'])
        metric_columns.append('num_layers')
    if 'data_points' in metrics:
        metrics_data.append(metrics['data_points'])
        metric_columns.append('data_points')
    
    # Create DataFrame and display
    if metrics_data:
        metrics_df = pd.DataFrame([metrics_data], columns=metric_columns)
        st.dataframe(metrics_df, hide_index=True)
    
    # Training Information
    st.write("Training Information:")
    training_info = {
        "Number of Features": metrics.get('n_features', ''),
        "Number of Samples": metrics.get('training_samples', ''),
        "Training Period": f"{metrics.get('training_period', {}).get('start', '')} to {metrics.get('training_period', {}).get('end', '')}"
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
    st.title("Base Model Training")
    
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
    
    # Get available tables
    available_tables = get_available_tables(db_path)
    if not available_tables:
        st.warning("No strategy tables found in the database.")
        return
    
    # Table Selection Section
    st.subheader("Select Tables for Training", help="Choose the tables you want to use for training")
    selected_tables = st.multiselect(
        label="Select Tables",
        options=available_tables,
        key="selected_tables",
        label_visibility="collapsed"
    )
    
    # Model Selection Section
    st.subheader("Select Model Types", help="Choose the type of model to train")
    model_types = st.multiselect(
        label="Select Model Types",
        options=['xgboost', 'decision_tree', 'random_forest', 'lstm'],
        default=['lstm'],
        key="model_types",
        label_visibility="collapsed"
    )
    
    # Training Type Section with Radio Buttons
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Training Type", help="Choose the type of training to perform")
        training_mode = st.radio(
            label="Training Mode",
            options=["single", "multi", "incremental"],
            horizontal=True,
            key="training_mode",
            label_visibility="collapsed"
        )
    
    with col2:
        st.subheader("Options")
        force_retrain = st.checkbox("Force Retrain", help="If checked, will perform a full retrain")
    
    # Show equivalent command
    if selected_tables and model_types:
        st.subheader("Equivalent Command")
        cmd = get_equivalent_command(training_mode, selected_tables, model_types, force_retrain)
        st.code(cmd, language="bash")
    
    # Training Button
    if st.button("Train Models", type="primary"):
        if not selected_tables:
            st.error("Please select at least one table for training.")
            return
            
        if not model_types:
            st.error("Please select at least one model type.")
            return
            
        try:
            # Show the command being executed
            st.info("Executing command:")
            st.code(get_equivalent_command(training_mode, selected_tables, model_types, force_retrain), language="bash")
            
            with st.spinner("Training model..."):
                if training_mode == "single":
                    if len(selected_tables) != 1:
                        st.error("Single mode requires exactly one table")
                        return
                    results = train_single_table(
                        table_name=selected_tables[0],
                        force_retrain=force_retrain,
                        model_types=model_types
                    )
                    
                elif training_mode == "multi":
                    if len(selected_tables) < 2:
                        st.error("Multi-table training requires at least two tables.")
                        return
                    results = train_multi_table(
                        table_names=selected_tables,
                        force_retrain=force_retrain,
                        model_types=model_types
                    )
                    
                else:  # incremental
                    if len(selected_tables) < 2:
                        st.error("Incremental mode requires at least two tables (base table and new tables)")
                        return
                    base_table = selected_tables[0]
                    new_tables = selected_tables[1:]
                    results = train_model_incrementally(
                        base_table=base_table,
                        new_tables=new_tables,
                        force_retrain=force_retrain,
                        model_types=model_types
                    )
                
                st.success("Training completed!")
                
                # Display results in a formatted way
                st.subheader("Training Results")
                st.json(results)
        except Exception as e:
            st.error(f"Error during training: {str(e)}") 