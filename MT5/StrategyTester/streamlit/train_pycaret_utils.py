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
import numpy as np
import plotly.graph_objects as go

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
        tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "🎯 Feature Analysis", "📈 Model-Specific Visualizations"])
        
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
        
        # Feature Analysis Tab
        with tab2:
            model_type = metrics.get('Model', '')
            
            # Check for feature importance in tree-based models
            if 'feature_importance_detailed' in metrics:
                importance_data = metrics['feature_importance_detailed']
                importance_df = pd.DataFrame({
                    'Feature': importance_data['feature'],
                    'Importance': importance_data['importance']
                }).sort_values('Importance', ascending=False)
                
                st.markdown("#### Feature Importance (Tree-based Model)")
                
                # Create a bar chart using plotly for better visualization
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h'
                ))
                
                fig.update_layout(
                    title='Feature Importance',
                    xaxis_title='Importance Score',
                    yaxis_title='Feature',
                    height=max(400, len(importance_df) * 20),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display feature importance as a table
                st.markdown("#### Feature Importance Details")
                styled_importance_df = importance_df.style.format({
                    'Importance': '{:.4f}'
                })
                st.dataframe(styled_importance_df, use_container_width=True)
            
            # Check for coefficients in linear models
            elif 'coefficients' in metrics:
                coef_data = metrics['coefficients']
                coef_df = pd.DataFrame({
                    'Feature': coef_data['feature'],
                    'Coefficient': coef_data['coefficient']
                })
                
                # Add absolute values for sorting
                coef_df['Absolute Coefficient'] = abs(coef_df['Coefficient'])
                coef_df = coef_df.sort_values('Absolute Coefficient', ascending=False)
                
                st.markdown("#### Feature Coefficients (Linear Model)")
                
                # Create a bar chart using plotly
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=coef_df['Coefficient'],
                    y=coef_df['Feature'],
                    orientation='h'
                ))
                
                fig.update_layout(
                    title='Feature Coefficients',
                    xaxis_title='Coefficient Value',
                    yaxis_title='Feature',
                    height=max(400, len(coef_df) * 20),
                    showlegend=False
                )
                
                # Add a vertical line at x=0 to better visualize positive/negative coefficients
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display coefficients as a table
                st.markdown("#### Coefficient Details")
                styled_coef_df = coef_df[['Feature', 'Coefficient']].style.format({
                    'Coefficient': '{:.4f}'
                })
                st.dataframe(styled_coef_df, use_container_width=True)
            
            else:
                st.info("No feature importance or coefficient information available for this model type.")

        # Model-Specific Visualizations Tab
        with tab3:
            if 'AllModels' in metrics:
                # Create tabs for each model
                model_names = list(metrics['AllModels'].keys())
                model_tabs = st.tabs(model_names)
                
                # Create visualizations for each model in its respective tab
                for model_tab, model_name in zip(model_tabs, model_names):
                    with model_tab:
                        model_metrics = metrics['AllModels'][model_name]
                        
                        # Get predictions and actual values from metrics if available
                        y_true = model_metrics.get('y_true', None)
                        y_pred = model_metrics.get('y_pred', None)
                        
                        if y_true is not None and y_pred is not None:
                            # Common visualizations for all models
                            st.markdown("#### Actual vs Predicted Values")
                            fig_scatter = create_scatter_plot(y_true, y_pred)
                            st.plotly_chart(fig_scatter, use_container_width=True)
                            
                            st.markdown("#### Residual Plot")
                            fig_residual = create_residual_plot(y_true, y_pred)
                            st.plotly_chart(fig_residual, use_container_width=True)
                            
                            # Model-specific visualizations
                            if model_name in ['Random Forest', 'XGBoost', 'LightGBM', 'CatBoost']:
                                st.markdown("#### Tree-based Model Visualizations")
                                if 'feature_importance_detailed' in model_metrics:
                                    fig_importance = create_detailed_importance_plot(
                                        model_metrics['feature_importance_detailed']
                                    )
                                    st.plotly_chart(fig_importance, use_container_width=True)
                            
                            elif model_name in ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet']:
                                st.markdown("#### Linear Model Visualizations")
                                if 'coefficients' in model_metrics:
                                    fig_coef = create_coefficient_plot(model_metrics['coefficients'])
                                    st.plotly_chart(fig_coef, use_container_width=True)
                            
                            elif model_name == 'K Neighbors Regressor':
                                st.markdown("#### K-Neighbors Visualizations")
                                if 'neighbor_distances' in model_metrics:
                                    fig_dist = create_neighbor_distance_plot(
                                        model_metrics['neighbor_distances']
                                    )
                                    st.plotly_chart(fig_dist, use_container_width=True)
                            
                            elif model_name == 'Support Vector Regression':
                                st.markdown("#### SVR Visualizations")
                                if 'support_vectors' in model_metrics:
                                    fig_sv = create_support_vector_plot(
                                        model_metrics['support_vectors']
                                    )
                                    st.plotly_chart(fig_sv, use_container_width=True)
                            
                            # Display metrics for this model
                            st.markdown("#### Model Metrics")
                            metrics_df = pd.DataFrame({
                                'Metric': ['MAE', 'RMSE', 'R²', 'MAPE', 'Directional Accuracy'],
                                'Value': [
                                    f"{model_metrics.get('MAE', 0):.4f}",
                                    f"{model_metrics.get('RMSE', 0):.4f}",
                                    f"{model_metrics.get('R2', 0):.4f}",
                                    f"{model_metrics.get('MAPE', 0):.2f}%",
                                    f"{model_metrics.get('DirectionalAccuracy', 0):.2f}%"
                                ]
                            })
                            st.dataframe(metrics_df, use_container_width=True)
            else:
                # Single model mode - use existing code
                model_type = metrics.get('Model', '')
                
                # Get predictions and actual values from metrics if available
                y_true = metrics.get('y_true', None)
                y_pred = metrics.get('y_pred', None)
                
                if y_true is not None and y_pred is not None:
                    # Common visualizations for all models
                    st.markdown("#### Actual vs Predicted Values")
                    fig_scatter = create_scatter_plot(y_true, y_pred)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    st.markdown("#### Residual Plot")
                    fig_residual = create_residual_plot(y_true, y_pred)
                    st.plotly_chart(fig_residual, use_container_width=True)
                    
                    # Model-specific visualizations
                    if model_type in ['Random Forest', 'XGBoost', 'LightGBM', 'CatBoost']:
                        st.markdown("#### Tree-based Model Visualizations")
                        if 'feature_importance_detailed' in metrics:
                            fig_importance = create_detailed_importance_plot(
                                metrics['feature_importance_detailed']
                            )
                            st.plotly_chart(fig_importance, use_container_width=True)
                    
                    elif model_type in ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet']:
                        st.markdown("#### Linear Model Visualizations")
                        if 'coefficients' in metrics:
                            fig_coef = create_coefficient_plot(metrics['coefficients'])
                            st.plotly_chart(fig_coef, use_container_width=True)
                    
                    elif model_type == 'K Neighbors Regressor':
                        st.markdown("#### K-Neighbors Visualizations")
                        if 'neighbor_distances' in metrics:
                            fig_dist = create_neighbor_distance_plot(
                                metrics['neighbor_distances']
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)
                    
                    elif model_type == 'Support Vector Regression':
                        st.markdown("#### SVR Visualizations")
                        if 'support_vectors' in metrics:
                            fig_sv = create_support_vector_plot(
                                metrics['support_vectors']
                            )
                            st.plotly_chart(fig_sv, use_container_width=True)
                    
                    # Display metrics for single model
                    st.markdown("#### Model Metrics")
                    metrics_df = pd.DataFrame({
                        'Metric': ['MAE', 'RMSE', 'R²', 'MAPE', 'Directional Accuracy'],
                        'Value': [
                            f"{metrics.get('MAE', 0):.4f}",
                            f"{metrics.get('RMSE', 0):.4f}",
                            f"{metrics.get('R2', 0):.4f}",
                            f"{metrics.get('MAPE', 0):.2f}%",
                            f"{metrics.get('DirectionalAccuracy', 0):.2f}%"
                        ]
                    })
                    st.dataframe(metrics_df, use_container_width=True)

def create_scatter_plot(y_true, y_pred):
    """Create scatter plot of actual vs predicted values using plotly"""
    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(
            size=8,
            color='blue',
            opacity=0.6
        )
    ))
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='Actual vs Predicted Values',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        showlegend=True
    )
    
    return fig

def create_residual_plot(y_true, y_pred):
    """Create residual plot using plotly"""
    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    residuals = y_true - y_pred
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(
            size=8,
            color='blue',
            opacity=0.6
        )
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title='Residual Plot',
        xaxis_title='Predicted Values',
        yaxis_title='Residuals',
        showlegend=True
    )
    
    return fig

def create_detailed_importance_plot(importance_data):
    """Create detailed feature importance plot for tree-based models"""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importance_data['importance'],
        y=importance_data['feature'],
        orientation='h'
    ))
    
    fig.update_layout(
        title='Feature Importance (Detailed)',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=max(400, len(importance_data['feature']) * 20)
    )
    
    return fig

def create_coefficient_plot(coefficients):
    """Create coefficient plot for linear models"""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=coefficients['coefficient'],
        y=coefficients['feature'],
        orientation='h'
    ))
    
    fig.update_layout(
        title='Model Coefficients',
        xaxis_title='Coefficient Value',
        yaxis_title='Feature',
        height=max(400, len(coefficients['feature']) * 20)
    )
    
    return fig

def create_neighbor_distance_plot(distances):
    """Create neighbor distance plot for KNN"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=distances,
        nbinsx=30,
        name='Distance Distribution'
    ))
    
    fig.update_layout(
        title='Neighbor Distance Distribution',
        xaxis_title='Distance',
        yaxis_title='Count'
    )
    
    return fig

def create_support_vector_plot(sv_data):
    """Create support vector visualization for SVR"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sv_data['x'],
        y=sv_data['y'],
        mode='markers',
        name='Support Vectors',
        marker=dict(
            size=10,
            color='red',
            symbol='circle-open'
        )
    ))
    
    fig.update_layout(
        title='Support Vectors',
        xaxis_title='Feature Space Dimension 1',
        yaxis_title='Feature Space Dimension 2'
    )
    
    return fig

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