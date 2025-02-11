import streamlit as st
import pandas as pd
from typing import List, Dict
import os
import sqlite3
from pycaret_model_trainer import PyCaretModelTrainer
from db_info import get_table_names, get_numeric_columns
from feature_config import get_feature_groups, get_all_features
import logging
from datetime import datetime
import queue
from logging.handlers import QueueHandler
import time

class StreamlitHandler(logging.Handler):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.logs = []
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.logs.append(msg)
            
            # Keep only last 1000 messages
            if len(self.logs) > 1000:
                self.logs = self.logs[-1000:]
            
            # Update the placeholder with all logs
            log_text = "\n".join(self.logs)
            self.placeholder.code(log_text)
            
        except Exception:
            self.handleError(record)

def setup_logging(placeholder):
    """Configure logging settings with Streamlit output"""
    # Create Streamlit handler
    streamlit_handler = StreamlitHandler(placeholder)
    streamlit_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    
    # Get root logger and add handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our custom handler
    root_logger.addHandler(streamlit_handler)
    
    return streamlit_handler

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

def get_model_types() -> List[str]:
    """Get available model types"""
    return [
        'Linear Regression',
        'Ridge',
        'Lasso',
        'ElasticNet',
        'Decision Tree',
        'Random Forest',
        'Extra Trees',
        'LightGBM',
        'XGBoost',
        'CatBoost',
        'AdaBoost',
        'Gradient Boosting',
        'Support Vector Regression',
        'K Neighbors Regressor',
        'Huber Regressor',
        'Bayesian Ridge'
    ]

def get_model_params(model_type: str) -> Dict:
    """Get model parameters based on model type"""
    if model_type == 'LightGBM':
        return {
            'n_estimators': st.number_input('Number of Estimators', 100, 2000, 1000, 100),
            'learning_rate': st.number_input('Learning Rate', 0.01, 0.5, 0.05, 0.01),
            'max_depth': st.slider('Max Depth', 3, 10, 8, 1),
            'subsample': st.slider('Subsample', 0.5, 1.0, 0.8, 0.1),
            'colsample_bytree': st.slider('Column Sample by Tree', 0.5, 1.0, 0.8, 0.1),
            'min_child_weight': st.number_input('Min Child Weight', 1, 10, 2, 1)
        }
    elif model_type == 'XGBoost':
        return {
            'max_depth': st.slider('Max Depth', 3, 10, 8, 1),
            'learning_rate': st.number_input('Learning Rate', 0.01, 0.5, 0.05, 0.01),
            'n_estimators': st.number_input('Number of Estimators', 100, 2000, 1000, 100),
            'subsample': st.slider('Subsample', 0.5, 1.0, 0.8, 0.1),
            'colsample_bytree': st.slider('Column Sample by Tree', 0.5, 1.0, 0.8, 0.1),
            'min_child_weight': st.number_input('Min Child Weight', 1, 10, 2, 1)
        }
    elif model_type == 'Random Forest':
        return {
            'n_estimators': st.number_input('Number of Estimators', 50, 500, 100, 50),
            'max_depth': st.slider('Max Depth', 3, 20, 10, 1),
            'min_samples_split': st.number_input('Min Samples Split', 2, 10, 2, 1),
            'min_samples_leaf': st.number_input('Min Samples Leaf', 1, 10, 1, 1),
            'max_features': st.selectbox('Max Features', ['auto', 'sqrt', 'log2'])
        }
    elif model_type == 'Decision Tree':
        return {
            'max_depth': st.slider('Max Depth', 3, 20, 10, 1),
            'min_samples_split': st.number_input('Min Samples Split', 2, 10, 2, 1),
            'min_samples_leaf': st.number_input('Min Samples Leaf', 1, 10, 1, 1),
            'criterion': st.selectbox('Criterion', ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'])
        }
    elif model_type == 'CatBoost':
        return {
            'iterations': st.number_input('Iterations', 100, 2000, 1000, 100),
            'learning_rate': st.number_input('Learning Rate', 0.01, 0.5, 0.05, 0.01),
            'depth': st.slider('Depth', 3, 10, 6, 1),
            'l2_leaf_reg': st.number_input('L2 Regularization', 1, 10, 3, 1)
        }
    elif model_type == 'AdaBoost':
        return {
            'n_estimators': st.number_input('Number of Estimators', 50, 500, 100, 50),
            'learning_rate': st.number_input('Learning Rate', 0.01, 2.0, 1.0, 0.1),
            'loss': st.selectbox('Loss Function', ['linear', 'square', 'exponential'])
        }
    elif model_type == 'Gradient Boosting':
        return {
            'n_estimators': st.number_input('Number of Estimators', 50, 500, 100, 50),
            'learning_rate': st.number_input('Learning Rate', 0.01, 0.5, 0.1, 0.01),
            'max_depth': st.slider('Max Depth', 3, 10, 3, 1),
            'subsample': st.slider('Subsample', 0.5, 1.0, 1.0, 0.1)
        }
    elif model_type == 'Support Vector Regression':
        return {
            'kernel': st.selectbox('Kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'C': st.number_input('C (Regularization)', 0.1, 10.0, 1.0, 0.1),
            'epsilon': st.number_input('Epsilon', 0.01, 1.0, 0.1, 0.01)
        }
    elif model_type == 'K Neighbors Regressor':
        return {
            'n_neighbors': st.number_input('Number of Neighbors', 1, 20, 5, 1),
            'weights': st.selectbox('Weight Function', ['uniform', 'distance']),
            'algorithm': st.selectbox('Algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        }
    elif model_type == 'ElasticNet':
        return {
            'alpha': st.number_input('Alpha (Regularization)', 0.01, 10.0, 1.0, 0.1),
            'l1_ratio': st.slider('L1 Ratio (0=Ridge, 1=Lasso)', 0.0, 1.0, 0.5, 0.1)
        }
    elif model_type == 'Huber Regressor':
        return {
            'epsilon': st.number_input('Epsilon', 1.1, 5.0, 1.35, 0.1),
            'alpha': st.number_input('Alpha (Regularization)', 0.0001, 1.0, 0.0001, 0.0001),
            'max_iter': st.number_input('Max Iterations', 100, 1000, 100, 100)
        }
    elif model_type == 'Bayesian Ridge':
        return {
            'n_iter': st.number_input('Number of Iterations', 100, 1000, 300, 100),
            'alpha_1': st.number_input('Alpha 1', 1e-6, 1e-4, 1e-6, 1e-6),
            'alpha_2': st.number_input('Alpha 2', 1e-6, 1e-4, 1e-6, 1e-6)
        }
    elif model_type == 'Extra Trees':
        return {
            'n_estimators': st.number_input('Number of Estimators', 50, 500, 100, 50),
            'max_depth': st.slider('Max Depth', 3, 20, 10, 1),
            'min_samples_split': st.number_input('Min Samples Split', 2, 10, 2, 1),
            'min_samples_leaf': st.number_input('Min Samples Leaf', 1, 10, 1, 1),
            'max_features': st.selectbox('Max Features', ['auto', 'sqrt', 'log2'])
        }
    else:  # Linear models (Ridge, Lasso)
        if model_type in ['Ridge', 'Lasso']:
            return {
                'alpha': st.number_input('Alpha (Regularization)', 0.01, 10.0, 1.0, 0.1),
                'max_iter': st.number_input('Max Iterations', 100, 2000, 1000, 100)
            }
        return {}  # Linear Regression has no parameters

def display_training_metrics(metrics: Dict):
    """Display training metrics in a formatted way"""
    if not metrics:
        return
    
    # Create a table-like display for metrics
    st.markdown("""
        <h4 style='color: #1565C0;'>Performance Metrics</h4>
    """, unsafe_allow_html=True)
    
    # Display metrics for the best model
    st.markdown("##### 🏆 Best Model Performance")
    best_metrics = {
        "📊 Model Type": metrics.get('Model', ''),
        "📉 MAE": f"{metrics.get('MAE', 0):.4f}",
        "📊 RMSE": f"{metrics.get('RMSE', 0):.4f}",
        "📈 R²": f"{metrics.get('R2', 0):.4f}",
        "💯 MAPE": f"{metrics.get('MAPE', 0):.2f}%",
        "🎯 Directional Accuracy": f"{metrics.get('DirectionalAccuracy', 0):.2f}%"
    }
    st.json(best_metrics)
    
    # Display feature importance if available
    if 'FeatureImportance' in metrics:
        st.markdown("##### 🎯 Feature Importance")
        importance_df = pd.DataFrame(
            list(metrics['FeatureImportance'].items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)
        
        st.bar_chart(importance_df.set_index('Feature')['Importance'])
    
    # Display all models' performance
    if 'AllModels' in metrics:
        st.markdown("##### 📊 All Models Performance")
        all_models_df = pd.DataFrame.from_dict(metrics['AllModels'], orient='index')
        st.dataframe(
            all_models_df.style.format({
                'MAE': '{:.4f}',
                'RMSE': '{:.4f}',
                'R2': '{:.4f}',
                'MAPE': '{:.2f}%',
                'DirectionalAccuracy': '{:.2f}%'
            }),
            use_container_width=True
        )

def get_equivalent_command(table_names: List[str], target_col: str, feature_cols: List[str], 
                         prediction_horizon: int, model_name: str) -> str:
    """Generate the equivalent command line command"""
    base_cmd = "python train_pycaret_models.py"
    tables_arg = f"--tables {' '.join(table_names)}"
    target_arg = f"--target {target_col}"
    features_arg = f"--features {' '.join(feature_cols)}"
    horizon_arg = f"--horizon {prediction_horizon}"
    model_arg = f"--model-name {model_name}" if model_name else ""
    
    return f"{base_cmd} {tables_arg} {target_arg} {features_arg} {horizon_arg} {model_arg}".strip()

def train_pycaret_models_page():
    """Streamlit page for training PyCaret models"""
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
    models_dir = os.path.join(current_dir, 'models')
    
    # Get available tables with detailed information
    available_tables = get_available_tables(db_path)
    if not available_tables:
        st.warning("No strategy tables found in the database.")
        return

    # Create main left-right layout
    left_col, right_col = st.columns([1, 1], gap="large")

    # Left Column - Configuration
    with left_col:
        st.markdown("""
            <p style='color: #666; margin: 0; font-size: 0.9em;'>Configure your AutoML training parameters</p>
            <hr style='margin: 0.2em 0 0.7em 0;'>
        """, unsafe_allow_html=True)
        
        # Table Selection Section
        st.markdown("##### 📊 Select Tables for Training")
        
        # Initialize session state variables if they don't exist
        if 'selected_tables' not in st.session_state:
            st.session_state['selected_tables'] = []
        if 'table_selections' not in st.session_state:
            st.session_state['table_selections'] = {}
        
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
        
        # Create a DataFrame for better visualization of table information
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
        
        table_df = pd.DataFrame(table_data)
        
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
            use_automl = st.radio(
                "Training Mode",
                ["AutoML (Try all models)", "Single Model"],
                help="Choose whether to try all models or train a specific one"
            )
            
            if use_automl:
                st.info("🤖 AutoML mode will try all models and select the best one")
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
            model_name = st.text_input(
                "Model Name",
                value=f"pycaret_model_{datetime.now().strftime('%Y%m%d_%H%M')}",
                help="Name for the trained model"
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
                        # Create a placeholder for live output in the right column
                        with right_col:
                            st.markdown("##### 📝 Training Progress")
                            output_placeholder = st.empty()
                            
                            # Setup logging with Streamlit output
                            streamlit_handler = setup_logging(output_placeholder)
                            
                            try:
                                with st.spinner("Training model..."):
                                    trainer = PyCaretModelTrainer(db_path, models_dir)
                                    
                                    # Prepare model parameters
                                    training_params = {}
                                    
                                    if not use_automl:
                                        training_params['model_type'] = model_type
                                        if model_params:
                                            training_params['model_params'] = {
                                                **model_params,
                                                'cv': cv_folds  # Include CV folds in model parameters
                                            }
                                    else:
                                        training_params['model_params'] = {
                                            'cv': cv_folds  # Include CV folds for AutoML
                                        }
                                    
                                    model_dir, metrics = trainer.train_and_save(
                                        table_names=st.session_state['selected_tables'],
                                        target_col=target_col,
                                        feature_cols=selected_features,
                                        prediction_horizon=prediction_horizon,
                                        model_name=model_name,
                                        **training_params
                                    )
                                    
                                    # Display success message
                                    st.success(f"✨ Model trained successfully and saved to {model_dir}")
                                    
                                    # Display metrics below the output
                                    display_training_metrics(metrics)
                            
                            finally:
                                # Clean up logging handler
                                root_logger = logging.getLogger()
                                root_logger.removeHandler(streamlit_handler)
                                
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")
                        logging.error(f"Training error: {str(e)}", exc_info=True)
    
    # Right Column - Results and Visualization
    with right_col:
        st.markdown("""
            <p style='color: #666; margin: 0; font-size: 0.9em;'>Training Results and Model Performance</p>
            <hr style='margin: 0.2em 0 0.7em 0;'>
        """, unsafe_allow_html=True)
        
        if not st.session_state['selected_tables']:
            st.info("👈 Please select tables and configure your model on the left to start training.")

if __name__ == "__main__":
    train_pycaret_models_page() 