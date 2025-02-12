import streamlit as st
import pandas as pd
import os
from pycaret_model_trainer import PyCaretModelTrainer
from db_info import get_table_names, get_numeric_columns
from feature_config import get_feature_groups, get_all_features
import logging
from datetime import datetime
from train_pycaret_utils import (
    initialize_session_state,
    check_stop_clicked,
    on_stop_click,
    setup_logging,
    get_available_tables,
    get_model_types,
    get_model_params,
    display_training_metrics,
    get_equivalent_command,
    TrainingInterrupt
)
from typing import Optional

def generate_model_name(model_type: str, training_type: str, timestamp: Optional[str] = None) -> str:
    """Generate consistent model name
    
    Args:
        model_type: Type of model (e.g., 'xgboost', 'decision_tree')
        training_type: Type of training ('single', 'multi', 'incremental', 'base')
        timestamp: Optional timestamp, will generate if None
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"pycaret-{model_type}_{training_type}_{timestamp}"

def train_pycaret_models_pagev2():
    """Streamlit page for training PyCaret models"""
    # Initialize session state
    initialize_session_state()
    
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
    models_dir = os.path.join(current_dir, 'models')
    
    # Get available tables with detailed information
    available_tables = get_available_tables(db_path)
    
    # Create main left-right layout
    left_col, right_col = st.columns([1, 1], gap="large")

    # Create a single container for the right column that persists
    right_container = right_col.container()

    # Right Column - Results and Visualization
    with right_container:
        st.markdown("""
            <p style='color: #666; margin: 0; font-size: 0.9em;'>Training Results and Model Performance</p>
            <hr style='margin: 0.2em 0 0.7em 0;'>
        """, unsafe_allow_html=True)
        
        # Create placeholders for different sections
        status_placeholder = st.empty()  # For info/warning messages
        training_progress = st.empty()  # For training logs
        stop_button_placeholder = st.empty()  # For stop button
        metrics_placeholder = st.empty()  # For metrics and results
        
        # Show stop message if exists
        if st.session_state.get('stop_message'):
            status_placeholder.warning(st.session_state['stop_message'])
        elif not st.session_state['selected_tables']:
            status_placeholder.info("👈 Please select tables and configure your model on the left to start training.")

        # Show training logs if they exist
        if st.session_state.get('training_logs'):
            with training_progress:
                st.markdown("##### 📝 Training Progress")
                st.code("\n".join(st.session_state['training_logs']))

    # Left Column - Configuration
    with left_col:
        st.markdown("""
            <p style='color: #666; margin: 0; font-size: 0.9em;'>Configure your AutoML training parameters</p>
            <hr style='margin: 0.2em 0 0.7em 0;'>
        """, unsafe_allow_html=True)
        
        if not available_tables and not st.session_state.get('stop_clicked', False):
            st.warning("No strategy tables found in the database.")
            return

        # Table Selection Section
        st.markdown("##### 📊 Select Tables for Training")
        
        # Initialize session state variables if they don't exist
        if 'selected_tables' not in st.session_state:
            st.session_state['selected_tables'] = []
        if 'table_selections' not in st.session_state:
            st.session_state['table_selections'] = {}
        if 'table_data' not in st.session_state:
            st.session_state['table_data'] = []
        
        def on_table_selection_change():
            """Callback to handle table selection changes"""
            edited_rows = st.session_state['table_editor']['edited_rows']
            for idx, changes in edited_rows.items():
                if '🔍 Select' in changes:
                    table_name = table_df.iloc[idx]['Table Name']
                    st.session_state['table_selections'][table_name] = changes['🔍 Select']
            
            # Update selected tables list
            st.session_state['selected_tables'] = [
                name for name, is_selected in st.session_state['table_selections'].items() 
                if is_selected
            ]
        
        # Create or use existing table data
        if not st.session_state['table_data']:
            table_data = []
            for t in available_tables:
                # Use the stored selection state or default to False
                is_selected = st.session_state['table_selections'].get(t['name'], False)
                table_data.append({
                    '🔍 Select': is_selected,
                    'Table Name': t['name'],
                    'Date Range': t['date_range'],
                    'Rows': t['total_rows'],
                    'Symbols': ', '.join(t['symbols'])
                })
            st.session_state['table_data'] = table_data
        
        table_df = pd.DataFrame(st.session_state['table_data'])
        
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
            key='table_editor',
            on_change=on_table_selection_change
        )

        # Model Configuration Section
        if st.session_state['selected_tables']:
            st.markdown("##### 🎯 Model Configuration")
            
            # Get numeric columns for the first selected table
            first_table = st.session_state['selected_tables'][0]
            numeric_cols = get_numeric_columns(db_path, first_table)
            
            # Target column selection
            target_col = st.selectbox(
                "Select target column",
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
                                key=f"feature_{feature}"
                            ):
                                selected_features.append(feature)
            
            # Additional numeric columns not in predefined groups
            other_numeric_cols = [col for col in numeric_cols if col not in all_features and col != target_col]
            if other_numeric_cols:
                with st.expander("Additional Numeric Features"):
                    for col in other_numeric_cols:
                        if st.checkbox(col, value=False, key=f"feature_{col}"):
                            selected_features.append(col)
            
            # Model Training Parameters
            st.markdown("##### ⚙️ Training Parameters")
            
            # Model Selection
            st.markdown("**Model Selection:**")
            training_mode = st.radio(
                "Training Mode",
                ["AutoML (Try all models)", "Single Model"],
                help="Choose whether to try all models or train a specific one"
            )
            
            if training_mode == "AutoML (Try all models)":
                st.info("🤖 Select which models to include in AutoML")
                
                # Option to select all models
                use_all_models = st.checkbox("Use All Available Models", value=True, 
                                           help="Select this to use all available models")
                
                # If not using all models, show multi-select
                if not use_all_models:
                    selected_models = st.multiselect(
                        "Select Models to Include",
                        options=get_model_types(),
                        default=['LightGBM', 'XGBoost', 'Random Forest'],
                        help="Choose which models to include in the AutoML process"
                    )
                    if not selected_models:
                        st.warning("⚠️ Please select at least one model")
                    else:
                        # Show parameters for each selected model
                        st.markdown("**Model Parameters:**")
                        model_specific_params = {}
                        for model in selected_models:
                            with st.expander(f"{model} Parameters"):
                                model_specific_params[model] = get_model_params(model)
                
                model_type = "automl"  # Set default model type for AutoML
            else:
                model_type = st.selectbox(
                    "Select Model",
                    options=get_model_types(),
                    help="Choose a specific model to train"
                )
                
                # Show model-specific parameters
                st.markdown("**Model Parameters:**")
                model_params = get_model_params(model_type)
            
            # Cross-validation settings
            st.markdown("**Cross-validation Settings:**")
            cv_folds = st.number_input(
                "Number of CV Folds",
                min_value=2,
                max_value=10,
                value=5,
                help="Number of folds for cross-validation"
            )
            
            # Prediction horizon
            st.markdown("**Prediction Settings:**")
            prediction_horizon = st.number_input(
                "Prediction Horizon (bars)",
                min_value=1,
                max_value=100,
                value=1,
                help="Number of bars ahead to predict"
            )
            
            # Model name
            if training_mode == "AutoML (Try all models)":
                model_type = "automl"
            model_name = generate_model_name(
                model_type=model_type,
                training_type="single",
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
            )

            # Display equivalent command
            if selected_features:
                st.markdown("##### 💻 Equivalent Command")
                cmd = get_equivalent_command(
                    st.session_state['selected_tables'],
                    target_col,
                    selected_features,
                    prediction_horizon,
                    model_name
                )
                st.code(cmd, language='bash')
                
                # Training button
                if st.button("🚀 Train Model", type="primary"):
                    try:
                        # Clear right side content and reset session state
                        st.session_state['stop_clicked'] = False
                        st.session_state['training_logs'] = []
                        st.session_state['stop_message'] = None
                        
                        # Clear all placeholders
                        status_placeholder.empty()
                        training_progress.empty()
                        stop_button_placeholder.empty()
                        metrics_placeholder.empty()
                        
                        # Show stop button
                        with stop_button_placeholder:
                            st.button("🛑 Stop Training", 
                                    on_click=on_stop_click,
                                    key="stop_training",
                                    help="Click to stop the training process",
                                    type="secondary")
                        
                        # Setup logging with Streamlit output
                        streamlit_handler = setup_logging(training_progress)
                        
                        try:
                            with st.spinner("Training model..."):
                                trainer = PyCaretModelTrainer(db_path, models_dir)
                                
                                # Prepare model parameters
                                training_params = {}
                                
                                if training_mode != "AutoML (Try all models)":
                                    training_params['model_params'] = {
                                        'model_type': model_type,  # Include model type in model_params
                                        'cv': cv_folds,  # Include CV folds in model parameters
                                    }
                                    if model_params:
                                        training_params['model_params'].update(model_params)  # Add other model parameters
                                else:
                                    training_params['model_params'] = {
                                        'cv': cv_folds,  # Include CV folds for AutoML
                                        'model_type': 'automl'  # Explicitly set model type for AutoML
                                    }
                                    # Add selected models for AutoML
                                    if not use_all_models:
                                        if not selected_models:
                                            st.error("Please select at least one model for AutoML")
                                            return
                                        training_params['model_params']['selected_models'] = selected_models
                                        # Add model-specific parameters if configured
                                        if 'model_specific_params' in locals():
                                            training_params['model_params']['model_specific_params'] = model_specific_params
                                        logging.info(f"Selected models for AutoML: {selected_models}")
                                
                                # Check for stop before starting training
                                if check_stop_clicked():
                                    raise TrainingInterrupt("Training stopped by user before starting")
                                
                                model_dir, metrics = trainer.train_and_save(
                                    table_names=st.session_state['selected_tables'],
                                    target_col=target_col,
                                    feature_cols=selected_features,
                                    prediction_horizon=prediction_horizon,
                                    model_name=model_name,
                                    **training_params
                                )
                                
                                # Display success message only if not stopped
                                if not check_stop_clicked():
                                    status_placeholder.success(f"✨ Model trained successfully and saved to {model_dir}")
                                    # Clear the stop button
                                    stop_button_placeholder.empty()
                                    # Display metrics below the output
                                    with metrics_placeholder:
                                        display_training_metrics(metrics)
                            
                        except TrainingInterrupt:
                            if 'stop_message' in st.session_state and st.session_state['stop_message']:
                                status_placeholder.warning(st.session_state['stop_message'])
                            # Clear the stop button
                            stop_button_placeholder.empty()
                            # Clean up any partial training artifacts if needed
                            if 'model_dir' in locals():
                                try:
                                    # Log cleanup attempt
                                    logging.info(f"Cleaning up partial training artifacts in {model_dir}")
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
    train_pycaret_models_pagev2()
