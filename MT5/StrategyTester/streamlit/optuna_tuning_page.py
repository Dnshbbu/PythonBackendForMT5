import streamlit as st
import pandas as pd
import os
from datetime import datetime
from optuna_model_trainer import OptunaModelTrainer
from db_info import get_table_names, get_numeric_columns
from feature_config import get_feature_groups, get_all_features
import logging
from train_pycaret_utils import (
    check_stop_clicked,
    on_stop_click,
    setup_logging,
    get_available_tables,
    TrainingInterrupt
)
import json
import numpy as np

def initialize_optuna_session_state():
    """Initialize session state variables for optuna page"""
    if 'optuna_selected_tables' not in st.session_state:
        st.session_state['optuna_selected_tables'] = []
    if 'optuna_table_selections' not in st.session_state:
        st.session_state['optuna_table_selections'] = {}
    if 'optuna_table_data' not in st.session_state:
        st.session_state['optuna_table_data'] = []
    if 'optuna_stop_clicked' not in st.session_state:
        st.session_state['optuna_stop_clicked'] = False
    if 'optuna_stop_message' not in st.session_state:
        st.session_state['optuna_stop_message'] = None
    if 'optuna_previous_selection' not in st.session_state:
        st.session_state['optuna_previous_selection'] = {
            'tables': [],
            'model_type': None
        }
    # Add prediction-specific state variables
    if 'optuna_pred_selected_tables' not in st.session_state:
        st.session_state['optuna_pred_selected_tables'] = []
    if 'optuna_pred_table_selections' not in st.session_state:
        st.session_state['optuna_pred_table_selections'] = {}
    if 'optuna_pred_table_data' not in st.session_state:
        st.session_state['optuna_pred_table_data'] = []

def clear_optuna_outputs():
    """Clear all optimization outputs"""
    st.session_state['optuna_stop_clicked'] = False
    st.session_state['optuna_stop_message'] = None

def on_optuna_table_selection_change():
    """Callback to handle table selection changes"""
    edited_rows = st.session_state['optuna_table_editor']['edited_rows']
    current_tables = []
    
    table_data = pd.DataFrame(st.session_state['optuna_table_data'])
    
    for idx, changes in edited_rows.items():
        if '🔍 Select' in changes:
            table_name = table_data.iloc[idx]['Table Name']
            st.session_state['optuna_table_selections'][table_name] = changes['🔍 Select']
            if changes['🔍 Select']:
                current_tables.append(table_name)
    
    st.session_state['optuna_selected_tables'] = [
        name for name, is_selected in st.session_state['optuna_table_selections'].items() 
        if is_selected
    ]
    
    if set(current_tables) != set(st.session_state['optuna_previous_selection']['tables']):
        clear_optuna_outputs()
        st.session_state['optuna_previous_selection']['tables'] = current_tables.copy()

def on_optuna_model_change():
    """Callback for model change"""
    current_model = st.session_state.get('optuna_model_selector')
    
    if current_model != st.session_state['optuna_previous_selection']['model_type']:
        clear_optuna_outputs()
        st.session_state['optuna_previous_selection']['model_type'] = current_model

def on_optuna_pred_table_selection_change():
    """Callback to handle prediction table selection changes"""
    edited_df = st.session_state['optuna_pred_table_editor']
    current_tables = []
    
    # Get the edited DataFrame
    if edited_df is not None and isinstance(edited_df, pd.DataFrame):
        # Process each row in the edited DataFrame
        for idx, row in edited_df.iterrows():
            if row['🔍 Select']:
                table_name = row['Table Name']
                current_tables.append(table_name)
                st.session_state['optuna_pred_table_selections'][table_name] = True
            else:
                table_name = row['Table Name']
                st.session_state['optuna_pred_table_selections'][table_name] = False
    
    # Update selected tables list
    st.session_state['optuna_pred_selected_tables'] = current_tables

def display_prediction_results(results_df: pd.DataFrame, model_name: str):
    """Display prediction results in a formatted way"""
    st.markdown("##### 📊 Prediction Results")
    
    # Key metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", len(results_df))
    with col2:
        st.metric("Mean Prediction", f"{results_df['Predicted_Value'].mean():.4f}")
    with col3:
        if 'Price' in results_df.columns:
            error = results_df['Predicted_Value'] - results_df['Price']
            st.metric("Mean Absolute Error", f"{error.abs().mean():.4f}")
    
    # Detailed results in tabs
    tabs = st.tabs(["Summary", "Detailed Results", "Visualization"])
    
    with tabs[0]:
        st.markdown("#### Summary Statistics")
        summary_stats = {
            "Model Used": model_name,
            "Data Points": len(results_df),
            "Time Range": f"{results_df['Date'].min()} to {results_df['Date'].max()}",
            "Mean Prediction": f"{results_df['Predicted_Value'].mean():.4f}",
            "Min Prediction": f"{results_df['Predicted_Value'].min():.4f}",
            "Max Prediction": f"{results_df['Predicted_Value'].max():.4f}"
        }
        if 'Price' in results_df.columns:
            summary_stats.update({
                "Mean Actual": f"{results_df['Price'].mean():.4f}",
                "Mean Error": f"{error.mean():.4f}",
                "RMSE": f"{(error ** 2).mean() ** 0.5:.4f}"
            })
        st.json(summary_stats)
    
    with tabs[1]:
        st.markdown("#### Detailed Results")
        st.dataframe(results_df)
        
        # Add download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name=f"optuna_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with tabs[2]:
        st.markdown("#### Visualization")
        if 'Price' in results_df.columns:
            # Plot actual vs predicted
            plot_df = pd.DataFrame({
                'Actual': results_df['Price'],
                'Predicted': results_df['Predicted_Value']
            }, index=pd.to_datetime(results_df['Date'] + ' ' + results_df['Time']))
            st.line_chart(plot_df)
            
            # Plot error distribution
            st.markdown("#### Error Distribution")
            st.line_chart(error)
        else:
            # Only plot predictions
            plot_df = pd.DataFrame({
                'Predicted': results_df['Predicted_Value']
            }, index=pd.to_datetime(results_df['Date'] + ' ' + results_df['Time']))
            st.line_chart(plot_df)

def optuna_tuning_page():
    """Streamlit page for Optuna hyperparameter optimization"""
    st.title("🎯 Optuna Hyperparameter Optimization")
    
    # Initialize session state
    initialize_optuna_session_state()
    
    # Create tabs for Training and Prediction
    train_tab, predict_tab = st.tabs(["🎯 Model Training", "🔮 Model Prediction"])
    
    with train_tab:
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        models_dir = os.path.join(current_dir, 'models')
        
        # Get available tables
        available_tables = get_available_tables(db_path)
        
        # Create main left-right layout
        left_col, right_col = st.columns([1, 1], gap="large")
        
        # Right Column - Results and Visualization
        right_container = right_col.container()
        
        with right_container:
            st.markdown("""
                <p style='color: #666; margin: 0; font-size: 0.9em;'>Optimization Results and Model Performance</p>
                <hr style='margin: 0.2em 0 0.7em 0;'>
            """, unsafe_allow_html=True)
            
            status_placeholder = st.empty()
            progress_placeholder = st.empty()
            stop_button_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            if not st.session_state.get('optuna_selected_tables'):
                status_placeholder.info("👈 Please select tables and configure your model on the left to start optimization.")
        
        # Left Column - Configuration
        with left_col:
            st.markdown("""
                <p style='color: #666; margin: 0; font-size: 0.9em;'>Configure Optuna Optimization Parameters</p>
                <hr style='margin: 0.2em 0 0.7em 0;'>
            """, unsafe_allow_html=True)
            
            if not available_tables:
                st.warning("No strategy tables found in the database.")
                return

            # Table Selection Section
            st.markdown("##### 📊 Select Tables for Training")
            
            if not st.session_state['optuna_table_data']:
                table_data = []
                for t in available_tables:
                    is_selected = st.session_state['optuna_table_selections'].get(t['name'], False)
                    table_data.append({
                        '🔍 Select': is_selected,
                        'Table Name': t['name'],
                        'Date Range': t['date_range'],
                        'Rows': t['total_rows'],
                        'Symbols': ', '.join(t['symbols'])
                    })
                st.session_state['optuna_table_data'] = table_data
            
            table_df = pd.DataFrame(st.session_state['optuna_table_data'])
            
            edited_df = st.data_editor(
                table_df,
                hide_index=True,
                column_config={
                    '🔍 Select': st.column_config.CheckboxColumn(
                        "Select",
                        help="Select tables for optimization",
                        default=False
                    )
                },
                key='optuna_table_editor',
                on_change=on_optuna_table_selection_change
            )

            if st.session_state['optuna_selected_tables']:
                # Get numeric columns for the first selected table
                first_table = st.session_state['optuna_selected_tables'][0]
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
                        help=f"Select all {group_name} features",
                        key=f"optuna_{group_name}"
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
                                    key=f"optuna_feature_{feature}"
                                ):
                                    selected_features.append(feature)
                
                # Additional numeric columns
                other_numeric_cols = [col for col in numeric_cols if col not in all_features and col != target_col]
                if other_numeric_cols:
                    with st.expander("Additional Numeric Features"):
                        for col in other_numeric_cols:
                            if st.checkbox(col, value=False, key=f"optuna_feature_{col}"):
                                selected_features.append(col)
                
                # Price Features Configuration
                st.markdown("##### 🕒 Price Features Configuration")
                use_price_features = st.checkbox(
                    "Use Current and Previous Prices as Features",
                    value=True,
                    help="Include current and previous price values to improve prediction",
                    key="optuna_use_price_features"
                )
                
                if use_price_features:
                    n_lags = st.number_input(
                        "Number of Previous Prices to Use",
                        min_value=1,
                        max_value=10,
                        value=3,
                        help="Number of previous price values to include as features",
                        key="optuna_n_lags"
                    )
                else:
                    n_lags = 0
                
                # Model Selection and Optimization Settings
                st.markdown("##### 🎯 Model Selection")
                model_type = st.selectbox(
                    "Select Model to Optimize",
                    options=['lightgbm', 'xgboost', 'random_forest', 'gradient_boosting', 
                            'elastic_net', 'svr', 'knn'],
                    help="Choose which model to optimize with Optuna",
                    key='optuna_model_selector',
                    on_change=on_optuna_model_change
                )
                
                # Optimization Parameters
                st.markdown("##### ⚙️ Optimization Settings")
                n_trials = st.number_input(
                    "Number of Trials",
                    min_value=10,
                    max_value=1000,
                    value=50,
                    help="Number of optimization trials to run",
                    key="optuna_n_trials"
                )
                
                timeout = st.number_input(
                    "Optimization Timeout (seconds)",
                    min_value=None,
                    value=None,
                    help="Maximum time for optimization (optional)",
                    key="optuna_timeout"
                )
                
                # Prediction Settings
                st.markdown("##### 📈 Prediction Settings")
                prediction_horizon = st.number_input(
                    "Prediction Horizon (bars)",
                    min_value=1,
                    max_value=100,
                    value=1,
                    help="Number of bars ahead to predict",
                    key="optuna_prediction_horizon"
                )
                
                # Display equivalent command
                st.markdown("##### 💻 Equivalent Command")
                optuna_cmd = f"""python train_optuna_cli.py \\
    --table_names {' '.join(st.session_state['optuna_selected_tables'])} \\
    --target_col {target_col} \\
    --feature_groups {' '.join(group for group, selected in selected_groups.items() if selected)} \\
    --model_type {model_type} \\
    --n_trials {n_trials} \\
    {"--timeout " + str(timeout) if timeout else ""} \\
    {"--use_price_features" if use_price_features else ""}"""
                
                st.code(optuna_cmd, language='bash')
                
                # Start Optimization button
                if st.button("🚀 Start Optimization", type="primary"):
                    try:
                        # Clear right side content
                        status_placeholder.empty()
                        progress_placeholder.empty()
                        metrics_placeholder.empty()
                        
                        # Show stop button
                        with stop_button_placeholder:
                            st.button("🛑 Stop Optimization", 
                                    on_click=on_stop_click,
                                    key="stop_optuna",
                                    help="Click to stop the optimization process",
                                    type="secondary")
                        
                        # Setup logging
                        streamlit_handler = setup_logging(progress_placeholder)
                        
                        try:
                            with st.spinner("Running Optuna optimization..."):
                                trainer = OptunaModelTrainer(db_path, models_dir)
                                
                                # Load and prepare data
                                df = trainer.load_data_from_db(st.session_state['optuna_selected_tables'][0])
                                X, y = trainer.prepare_features_target(
                                    df=df,
                                    target_col=target_col,
                                    feature_cols=selected_features,
                                    prediction_horizon=prediction_horizon,
                                    n_lags=n_lags if use_price_features else 0,
                                    use_price_features=use_price_features
                                )
                                
                                # Run Optuna optimization
                                model, metrics = trainer.train_with_optuna(
                                    X=X,
                                    y=y,
                                    model_type=model_type,
                                    n_trials=n_trials,
                                    timeout=timeout
                                )
                                
                                # Save model and display results
                                model_path = trainer.save_model_and_metadata(
                                    model=model,
                                    metrics=metrics,
                                    model_name=f"optuna_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                )
                                
                                status_placeholder.success(f"✨ Model optimized successfully and saved to {model_path}")
                                
                                # Display metrics
                                with metrics_placeholder:
                                    # First show the optimization progress
                                    if 'optimization_history' in metrics and 'trial_history' in metrics['optimization_history']:
                                        st.markdown("##### 📊 Optimization Progress")
                                        # Create an expandable section for trial details
                                        with st.expander("View Trial Details", expanded=True):
                                            # Display trial history in a scrollable container
                                            st.markdown(
                                                """<div style='height: 300px; overflow-y: scroll;'>""" +
                                                "<br>".join(metrics['optimization_history']['trial_history']) +
                                                "</div>",
                                                unsafe_allow_html=True
                                            )
                                        
                                        # Show optimization history plot
                                        st.markdown("##### 📈 Optimization History")
                                        history_df = pd.DataFrame({
                                            'Trial': range(len(metrics['optimization_history']['values'])),
                                            'Score': metrics['optimization_history']['values']
                                        })
                                        st.line_chart(history_df.set_index('Trial'))
                                    
                                    # Show final metrics in a tabbed view
                                    tab1, tab2 = st.tabs(["📊 Model Metrics", "📈 Training History"])
                                    
                                    # Model Metrics Tab
                                    with tab1:
                                        st.markdown("#### Model Performance Metrics")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric(
                                                "MAE",
                                                f"{metrics['MAE']:.4f}",
                                                help="Mean Absolute Error"
                                            )
                                            st.metric(
                                                "RMSE",
                                                f"{metrics['RMSE']:.4f}",
                                                help="Root Mean Square Error"
                                            )
                                        with col2:
                                            st.metric(
                                                "R² Score",
                                                f"{metrics['R2']:.4f}",
                                                help="R-squared Score (Coefficient of Determination)"
                                            )
                                            st.metric(
                                                "Directional Accuracy",
                                                f"{metrics['DirectionalAccuracy']:.2f}%",
                                                help="Percentage of correct price movement predictions"
                                            )
                                        
                                        # Display best parameters
                                        st.markdown("#### Best Parameters")
                                        st.json(metrics.get('best_params', {}))
                                    
                                    # Training History Tab
                                    with tab2:
                                        st.markdown("#### Optimization History")
                                        
                                        # Plot optimization progress
                                        st.markdown("##### Score Evolution")
                                        history_df = pd.DataFrame({
                                            'Trial': range(len(metrics['optimization_history']['values'])),
                                            'Score': metrics['optimization_history']['values']
                                        })
                                        
                                        # Add rolling mean to show trend
                                        history_df['Rolling Mean (10 trials)'] = history_df['Score'].rolling(
                                            window=min(10, len(history_df)),
                                            min_periods=1
                                        ).mean()
                                        
                                        # Plot both actual scores and rolling mean
                                        st.line_chart(history_df.set_index('Trial'))
                                        
                                        # Show parameter importance if available
                                        if 'parameter_importance' in metrics:
                                            st.markdown("##### Parameter Importance")
                                            param_importance_df = pd.DataFrame(
                                                metrics['parameter_importance']
                                            ).sort_values('importance', ascending=True)
                                            
                                            # Create a horizontal bar chart
                                            st.bar_chart(
                                                param_importance_df.set_index('parameter')['importance']
                                            )
                                        
                                        # Show trial statistics
                                        st.markdown("##### Trial Statistics")
                                        total_trials = len(metrics['optimization_history']['values'])
                                        completed_trials = len([t for t in metrics['optimization_history']['trial_history'] 
                                                             if "Pruned" not in t])
                                        pruned_trials = total_trials - completed_trials
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Total Trials", total_trials)
                                        with col2:
                                            st.metric("Completed Trials", completed_trials)
                                        with col3:
                                            st.metric("Pruned Trials", pruned_trials)
                                    
                                    # Add download button for the model and results
                                    st.markdown("#### 💾 Save Results")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        # Convert metrics to JSON for download
                                        metrics_json = json.dumps(metrics, indent=2)
                                        st.download_button(
                                            label="📥 Download Metrics JSON",
                                            data=metrics_json,
                                            file_name=f"optuna_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                            mime="application/json"
                                        )
                                    with col2:
                                        st.markdown(f"📁 Model saved at: `{model_path}`")
                        
                        except TrainingInterrupt:
                            status_placeholder.warning(st.session_state['optuna_stop_message'])
                        
                    except Exception as e:
                        status_placeholder.error(f"Error during optimization: {str(e)}")
                        logging.error(f"Optimization error: {str(e)}", exc_info=True)
                    
                    finally:
                        # Clean up
                        root_logger = logging.getLogger()
                        if 'streamlit_handler' in locals():
                            root_logger.removeHandler(streamlit_handler)
                        stop_button_placeholder.empty()

    with predict_tab:
        st.markdown("### 🔮 Make Predictions with Optimized Model")
        
        # Setup paths for prediction
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        models_dir = os.path.join(current_dir, 'models')
        
        # Create columns for prediction interface
        pred_col1, pred_col2 = st.columns([1, 1], gap="large")
        
        with pred_col1:
            st.markdown("""
                <p style='color: #666; margin: 0; font-size: 0.9em;'>Configure prediction parameters</p>
                <hr style='margin: 0.2em 0 0.7em 0;'>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📂 Load Model")
            
            # Scan for Optuna models
            optuna_models = []
            if os.path.exists(models_dir):
                for item in os.listdir(models_dir):
                    if item.startswith('optuna_'):
                        model_path = os.path.join(models_dir, item)
                        if os.path.isdir(model_path):
                            optuna_models.append(item)
            
            if not optuna_models:
                st.warning("No optimized models found. Please train a model first.")
                return
            
            # Model selection
            selected_model = st.selectbox(
                "Select Model",
                options=optuna_models,
                help="Choose an optimized model for prediction"
            )
            
            # Load model metadata
            model_path = os.path.join(models_dir, selected_model)
            try:
                metadata_path = os.path.join(model_path, 'metadata.json')
                if not os.path.exists(metadata_path):
                    st.warning(f"No metadata file found at {metadata_path}")
                    st.info("Please check if the model was trained correctly.")
                    return
                
                with open(metadata_path, 'r') as f:
                    model_metadata = json.load(f)
                
                # Display model information in a more structured way
                st.markdown("#### 📋 Model Information")
                
                # Basic model info
                st.markdown("##### Basic Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Model Type:** {model_metadata.get('model_type', 'Not specified')}")
                    st.markdown(f"**Target Column:** {model_metadata.get('target', 'Not specified')}")
                with col2:
                    st.markdown(f"**Prediction Horizon:** {model_metadata.get('prediction_horizon', 'Not specified')}")
                    st.markdown(f"**Number of Features:** {len(model_metadata.get('features', []))}")
                
                # Show features in an expander
                with st.expander("View Model Features", expanded=False):
                    features = model_metadata.get('features', [])
                    if features:
                        st.markdown("**Selected Features:**")
                        for feature in features:
                            st.markdown(f"- {feature}")
                    else:
                        st.info("No feature information available")
                
                # Show hyperparameters in an expander
                with st.expander("View Model Hyperparameters", expanded=False):
                    if 'best_params' in model_metadata:
                        st.json(model_metadata['best_params'])
                    else:
                        st.info("No hyperparameter information available")
                
                # Show performance metrics if available
                if 'metrics' in model_metadata:
                    with st.expander("View Model Performance Metrics", expanded=False):
                        metrics = model_metadata['metrics']
                        col1, col2 = st.columns(2)
                        with col1:
                            if 'MAE' in metrics:
                                st.metric("MAE", f"{metrics['MAE']:.4f}")
                            if 'RMSE' in metrics:
                                st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                        with col2:
                            if 'R2' in metrics:
                                st.metric("R² Score", f"{metrics['R2']:.4f}")
                            if 'DirectionalAccuracy' in metrics:
                                st.metric("Directional Accuracy", f"{metrics['DirectionalAccuracy']:.2f}%")
                
                # Table Selection Section
                st.markdown("#### 📊 Select Tables for Prediction")
                
                # Get available tables
                available_tables = get_available_tables(db_path)
                
                if not available_tables:
                    st.warning("No strategy tables found in the database.")
                    return
                
                # Create table selection interface
                table_data = []
                for t in available_tables:
                    is_selected = st.session_state.get('optuna_pred_table_selections', {}).get(t['name'], False)
                    table_data.append({
                        '🔍 Select': is_selected,
                        'Table Name': t['name'],
                        'Date Range': t['date_range'],
                        'Rows': t['total_rows'],
                        'Symbols': ', '.join(t['symbols'])
                    })
                
                table_df = pd.DataFrame(table_data)
                
                st.markdown("##### Select tables from the list below:")
                edited_df = st.data_editor(
                    table_df,
                    hide_index=True,
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
                            width="small"
                        ),
                        "Symbols": st.column_config.TextColumn(
                            "Symbols",
                            help="Trading symbols in the table",
                            width="medium"
                        )
                    },
                    key='optuna_pred_table_editor',
                    on_change=on_optuna_pred_table_selection_change
                )
                
                # Get selected tables from session state
                selected_tables = st.session_state['optuna_pred_selected_tables']
                
                if selected_tables:
                    st.success(f"Selected {len(selected_tables)} table(s) for prediction")
                    
                    # Add batch size option
                    batch_size = st.number_input(
                        "Batch Size",
                        min_value=100,
                        max_value=10000,
                        value=1000,
                        step=100,
                        help="Number of rows to process at once"
                    )
                    
                    # Make predictions button
                    if st.button("🔮 Make Predictions", type="primary", key="make_predictions_button"):
                        try:
                            # Create a progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Load the model and scaler
                            import joblib
                            model = joblib.load(os.path.join(model_path, 'model.pkl'))
                            scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
                            
                            # Initialize trainer for data loading
                            trainer = OptunaModelTrainer(db_path, models_dir)
                            
                            all_predictions = []
                            total_tables = len(selected_tables)
                            
                            # Process each selected table
                            for table_idx, table_name in enumerate(selected_tables):
                                status_text.text(f"Processing table {table_idx + 1}/{total_tables}: {table_name}")
                                
                                # Load and prepare data
                                df = trainer.load_data_from_db(table_name)
                                
                                # Update progress
                                progress = (table_idx) / total_tables
                                progress_bar.progress(progress)
                                
                                # Prepare features using the same configuration as training
                                X, _ = trainer.prepare_features_target(
                                    df=df,
                                    target_col=model_metadata.get('target', 'Price'),
                                    feature_cols=model_metadata['features'],
                                    prediction_horizon=model_metadata.get('prediction_horizon', 1),
                                    n_lags=model_metadata.get('n_lags', 0),
                                    use_price_features=model_metadata.get('use_price_features', False)
                                )
                                
                                # Process in batches
                                predictions = []
                                for i in range(0, len(X), batch_size):
                                    batch = X[i:i + batch_size]
                                    batch_scaled = scaler.transform(batch)
                                    batch_predictions = model['model'].predict(batch_scaled)
                                    predictions.extend(batch_predictions)
                                    
                                    # Update progress within table
                                    sub_progress = (table_idx + (i + len(batch)) / len(X)) / total_tables
                                    progress_bar.progress(sub_progress)
                                
                                # Create results DataFrame
                                results_df = df.copy()
                                results_df['Predicted_Value'] = predictions
                                results_df['Source_Table'] = table_name
                                
                                all_predictions.append(results_df)
                            
                            # Complete progress
                            progress_bar.progress(1.0)
                            status_text.text("Processing complete!")
                            
                            # Combine all predictions
                            final_results = pd.concat(all_predictions, axis=0)
                            
                            # Display results using the new display function
                            with pred_col2:
                                display_prediction_results(final_results, selected_model)
                        
                        except Exception as e:
                            st.error(f"Error making predictions: {str(e)}")
                            logging.error(f"Prediction error: {str(e)}", exc_info=True)
                
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                logging.error(f"Model loading error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    optuna_tuning_page() 