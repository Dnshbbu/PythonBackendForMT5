import streamlit as st
import pandas as pd
from typing import List, Dict
import os
import sqlite3
import logging
from datetime import datetime
import queue
from logging.handlers import QueueHandler
import time
import threading
import signal
import sys

class TrainingInterrupt(Exception):
    """Custom exception for interrupting the training process"""
    pass

class StreamlitHandler(logging.Handler):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        if 'training_logs' not in st.session_state:
            st.session_state['training_logs'] = []
        self.logs = st.session_state['training_logs']
    
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
            
            # Force a Streamlit rerun to update the UI
            st.session_state['training_logs'] = self.logs
            
            # Check for stop after each log message
            if check_stop_clicked():
                raise TrainingInterrupt("Training stopped by user")
            
        except TrainingInterrupt:
            raise
        except Exception:
            self.handleError(record)

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'selected_tables' not in st.session_state:
        st.session_state['selected_tables'] = []
    if 'training_logs' not in st.session_state:
        st.session_state['training_logs'] = []
    if 'stop_clicked' not in st.session_state:
        st.session_state['stop_clicked'] = False
    if 'stop_message' not in st.session_state:
        st.session_state['stop_message'] = None

def check_stop_clicked():
    """Check if stop button was clicked"""
    return st.session_state.get('stop_clicked', False)

def on_stop_click():
    """Callback for stop button click"""
    st.session_state['stop_clicked'] = True
    st.session_state['stop_message'] = "⚠️ Training was stopped by user"

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
    # Check if training was stopped
    if st.session_state.get('stop_clicked', False):
        return []
        
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
        if not st.session_state.get('stop_clicked', False):
            st.error(f"Error accessing database: {str(e)}")
        return []

def get_model_types() -> List[str]:
    """Get available model types"""
    models = [
        'Linear Regression',
        'Ridge',
        'Lasso',
        'ElasticNet',
        'LightGBM',
        'XGBoost',
        'Random Forest',
        'K Neighbors Regressor',
        'AdaBoost',
        'Gradient Boosting',
        'Support Vector Regression',
        'Huber Regressor',
        'Bayesian Ridge'
    ]
    
    # Add CatBoost if available
    try:
        from catboost import CatBoostRegressor
        models.append('CatBoost')
    except ImportError:
        pass
        
    return models

def get_model_params(model_type: str) -> Dict:
    """Get model parameters based on model type"""
    if model_type == 'LightGBM':
        return {
            'n_estimators': st.number_input('Number of Estimators', 100, 2000, 1000, 100, key=f"lgb_n_est_{model_type}"),
            'learning_rate': st.number_input('Learning Rate', 0.01, 0.5, 0.05, 0.01, key=f"lgb_lr_{model_type}"),
            'max_depth': st.slider('Max Depth', 3, 10, 8, 1, key=f"lgb_depth_{model_type}"),
            'subsample': st.slider('Subsample', 0.5, 1.0, 0.8, 0.1, key=f"lgb_ss_{model_type}"),
            'colsample_bytree': st.slider('Column Sample by Tree', 0.5, 1.0, 0.8, 0.1, key=f"lgb_cs_{model_type}"),
            'min_child_weight': st.number_input('Min Child Weight', 1, 10, 2, 1, key=f"lgb_mcw_{model_type}")
        }
    elif model_type == 'XGBoost':
        return {
            'max_depth': st.slider('Max Depth', 3, 10, 8, 1, key=f"xgb_depth_{model_type}"),
            'learning_rate': st.number_input('Learning Rate', 0.01, 0.5, 0.05, 0.01, key=f"xgb_lr_{model_type}"),
            'n_estimators': st.number_input('Number of Estimators', 100, 2000, 1000, 100, key=f"xgb_n_est_{model_type}"),
            'subsample': st.slider('Subsample', 0.5, 1.0, 0.8, 0.1, key=f"xgb_ss_{model_type}"),
            'colsample_bytree': st.slider('Column Sample by Tree', 0.5, 1.0, 0.8, 0.1, key=f"xgb_cs_{model_type}"),
            'min_child_weight': st.number_input('Min Child Weight', 1, 10, 2, 1, key=f"xgb_mcw_{model_type}")
        }
    elif model_type == 'Random Forest':
        return {
            'n_estimators': st.number_input('Number of Estimators', 50, 500, 100, 50, key=f"rf_n_est_{model_type}"),
            'max_depth': st.slider('Max Depth', 3, 20, 10, 1, key=f"rf_depth_{model_type}"),
            'min_samples_split': st.number_input('Min Samples Split', 2, 10, 2, 1, key=f"rf_mss_{model_type}"),
            'min_samples_leaf': st.number_input('Min Samples Leaf', 1, 10, 1, 1, key=f"rf_msl_{model_type}"),
            'max_features': st.selectbox('Max Features', ['auto', 'sqrt', 'log2'], key=f"rf_mf_{model_type}")
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
            'iterations': st.number_input('Iterations', 100, 2000, 1000, 100, key=f"cb_iter_{model_type}"),
            'learning_rate': st.number_input('Learning Rate', 0.01, 0.5, 0.05, 0.01, key=f"cb_lr_{model_type}"),
            'depth': st.slider('Depth', 3, 10, 6, 1, key=f"cb_depth_{model_type}"),
            'l2_leaf_reg': st.number_input('L2 Regularization', 1, 10, 3, 1, key=f"cb_l2_{model_type}")
        }
    elif model_type == 'AdaBoost':
        return {
            'n_estimators': st.number_input('Number of Estimators', 50, 500, 100, 50, key=f"ab_n_est_{model_type}"),
            'learning_rate': st.number_input('Learning Rate', 0.01, 2.0, 1.0, 0.1, key=f"ab_lr_{model_type}"),
            'loss': st.selectbox('Loss Function', ['linear', 'square', 'exponential'], key=f"ab_loss_{model_type}")
        }
    elif model_type == 'Gradient Boosting':
        return {
            'n_estimators': st.number_input('Number of Estimators', 50, 500, 100, 50, key=f"gb_n_est_{model_type}"),
            'learning_rate': st.number_input('Learning Rate', 0.01, 0.5, 0.1, 0.01, key=f"gb_lr_{model_type}"),
            'max_depth': st.slider('Max Depth', 3, 10, 3, 1, key=f"gb_depth_{model_type}"),
            'subsample': st.slider('Subsample', 0.5, 1.0, 1.0, 0.1, key=f"gb_ss_{model_type}")
        }
    elif model_type == 'Support Vector Regression':
        return {
            'kernel': st.selectbox('Kernel', ['linear', 'poly', 'rbf', 'sigmoid'], key=f"svr_k_{model_type}"),
            'C': st.number_input('C (Regularization)', 0.1, 10.0, 1.0, 0.1, key=f"svr_c_{model_type}"),
            'epsilon': st.number_input('Epsilon', 0.01, 1.0, 0.1, 0.01, key=f"svr_e_{model_type}")
        }
    elif model_type == 'K Neighbors Regressor':
        return {
            'n_neighbors': st.number_input('Number of Neighbors', 1, 20, 5, 1, key=f"knn_n_{model_type}"),
            'weights': st.selectbox('Weight Function', ['uniform', 'distance'], key=f"knn_w_{model_type}"),
            'algorithm': st.selectbox('Algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'], key=f"knn_algo_{model_type}")
        }
    elif model_type == 'ElasticNet':
        return {
            'alpha': st.number_input('Alpha (Regularization)', 0.01, 10.0, 1.0, 0.1, key=f"en_alpha_{model_type}"),
            'l1_ratio': st.slider('L1 Ratio (0=Ridge, 1=Lasso)', 0.0, 1.0, 0.5, 0.1, key=f"en_l1_{model_type}")
        }
    elif model_type == 'Huber Regressor':
        return {
            'epsilon': st.number_input('Epsilon', 1.1, 5.0, 1.35, 0.1, key=f"hr_e_{model_type}"),
            'alpha': st.number_input('Alpha (Regularization)', 0.0001, 1.0, 0.0001, 0.0001, key=f"hr_a_{model_type}"),
            'max_iter': st.number_input('Max Iterations', 100, 1000, 100, 100, key=f"hr_mi_{model_type}")
        }
    elif model_type == 'Bayesian Ridge':
        return {
            'n_iter': st.number_input('Number of Iterations', 100, 1000, 300, 100, key=f"br_ni_{model_type}"),
            'alpha_1': st.number_input('Alpha 1', 1e-6, 1e-4, 1e-6, 1e-6, key=f"br_a1_{model_type}"),
            'alpha_2': st.number_input('Alpha 2', 1e-6, 1e-4, 1e-6, 1e-6, key=f"br_a2_{model_type}")
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
                'alpha': st.number_input('Alpha (Regularization)', 0.01, 10.0, 1.0, 0.1, key=f"linear_alpha_{model_type}"),
                'max_iter': st.number_input('Max Iterations', 100, 2000, 1000, 100, key=f"linear_iter_{model_type}")
            }
        return {}  # Linear Regression has no parameters

def display_training_metrics(metrics: Dict):
    """Display training metrics in a formatted way"""
    if not metrics:
        return
    
    # Create main expander for all results
    with st.expander("🎯 Training Results", expanded=True):
        # Best Model Performance with enhanced visualization
        st.markdown("### 🏆 Best Model Performance")
        st.markdown("---")
        
        # Create three columns for best model metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Model Type",
                metrics.get('Model', ''),
                delta=None,
                help="The best performing model"
            )
            st.metric(
                "MAE",
                f"{metrics.get('MAE', 0):.4f}",
                delta=None,
                help="Mean Absolute Error"
            )
        
        with col2:
            st.metric(
                "RMSE",
                f"{metrics.get('RMSE', 0):.4f}",
                delta=None,
                help="Root Mean Square Error"
            )
            st.metric(
                "R²",
                f"{metrics.get('R2', 0):.4f}",
                delta=None,
                help="R-squared score"
            )
        
        with col3:
            st.metric(
                "MAPE",
                f"{metrics.get('MAPE', 0):.2f}%",
                delta=None,
                help="Mean Absolute Percentage Error"
            )
            st.metric(
                "Directional Accuracy",
                f"{metrics.get('DirectionalAccuracy', 0):.2f}%",
                delta=None,
                help="Percentage of correct direction predictions"
            )
        
        st.markdown("---")
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["📊 Model Comparison", "🎯 Feature Importance"])
        
        # Model Comparison Tab
        with tab1:
            if 'AllModels' in metrics:
                # Convert the dictionary to a DataFrame
                all_models_df = pd.DataFrame.from_dict(metrics['AllModels'], orient='index')
                
                # Ensure all required columns exist and in correct order
                required_cols = ['MAE', 'RMSE', 'R2', 'MAPE', 'DirectionalAccuracy']
                for col in required_cols:
                    if col not in all_models_df.columns:
                        all_models_df[col] = 0.0
                
                # Reorder columns and sort by MAE
                all_models_df = all_models_df[required_cols].sort_values('MAE')
                
                # Format the DataFrame without using background_gradient
                styled_df = all_models_df.style.format({
                    'MAE': '{:.4f}',
                    'RMSE': '{:.4f}',
                    'R2': '{:.4f}',
                    'MAPE': '{:.2f}%',
                    'DirectionalAccuracy': '{:.2f}%'
                })
                
                # Display the table
                st.dataframe(
                    styled_df,
                    hide_index=False,
                    use_container_width=True
                )
                
                # Add a bar chart comparison using Streamlit's native bar_chart
                st.bar_chart(
                    all_models_df[['MAE', 'RMSE']],
                    use_container_width=True
                )
        
        # Feature Importance Tab
        with tab2:
            if 'FeatureImportance' in metrics:
                # Create DataFrame for feature importance
                importance_df = pd.DataFrame(
                    list(metrics['FeatureImportance'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
                
                # Create a bar chart using Streamlit's native bar_chart
                st.bar_chart(
                    importance_df.set_index('Feature')['Importance'],
                    use_container_width=True
                )
                
                # Display feature importance as a table
                st.markdown("#### Feature Importance Details")
                styled_importance_df = importance_df.style.format({
                    'Importance': '{:.4f}'
                })
                
                st.dataframe(
                    styled_importance_df,
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