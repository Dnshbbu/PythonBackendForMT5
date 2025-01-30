"""
ml_analysis_page.py - Enhanced Machine Learning Analysis Page Implementation with XGBoost
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional, Any

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, accuracy_score, 
    precision_score, recall_score, f1_score
)

# Model imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.svm import SVR

import plotly.express as px
import plotly.graph_objects as go
from model_pipeline import ModelPipeline


def get_algorithm_params(algorithm: str) -> Dict:
    """Get algorithm-specific parameters from user input"""
    params = {}
    
    if algorithm == "xgboost":
        col1, col2 = st.columns(2)
        with col1:
            params['n_estimators'] = st.slider(
                "Number of Estimators",
                10, 500, 200
            )
            params['learning_rate'] = st.slider(
                "Learning Rate",
                0.01, 0.3, 0.05
            )
            params['max_depth'] = st.slider(
                "Max Depth",
                3, 10, 6
            )
        with col2:
            params['min_child_weight'] = st.slider(
                "Min Child Weight",
                1, 7, 2
            )
            params['subsample'] = st.slider(
                "Subsample",
                0.5, 1.0, 0.8
            )
            params['colsample_bytree'] = st.slider(
                "Colsample Bytree",
                0.5, 1.0, 0.8
            )
    elif algorithm in ["ridge", "lasso"]:
        params['alpha'] = st.slider(
            "Alpha",
            0.0, 1.0, 0.1
        )
    elif algorithm in ["random_forest", "gradient_boosting"]:
        params['n_estimators'] = st.slider(
            "Number of Estimators",
            10, 500, 100
        )
    
    return params

def run_regression(
    data: pd.DataFrame,
    target_col: str,
    algorithm: str,
    params: Dict
) -> Tuple[Dict[str, float], pd.DataFrame, Any, Any, np.ndarray, np.ndarray, np.ndarray]:
    """Enhanced regression analysis with XGBoost support"""
    # Prepare data
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    # Use RobustScaler for better handling of outliers
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.2, random_state=42
    )
    
    # Initialize model
    if algorithm == "xgboost":
        model = XGBRegressor(
            n_estimators=params.get('n_estimators', 200),
            learning_rate=params.get('learning_rate', 0.05),
            max_depth=params.get('max_depth', 6),
            min_child_weight=params.get('min_child_weight', 2),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            random_state=42
        )
    elif algorithm == "linear":
        model = LinearRegression()
    elif algorithm == "ridge":
        model = Ridge(alpha=params.get('alpha', 1.0))
    elif algorithm == "lasso":
        model = Lasso(alpha=params.get('alpha', 1.0))
    elif algorithm == "random_forest":
        model = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            random_state=42
        )
    elif algorithm == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=params.get('n_estimators', 100),
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Train model with early stopping for XGBoost
    if algorithm == "xgboost":
        eval_set = [(X_test, y_test.ravel())]
        model.set_params(callbacks=[xgb.callback.EarlyStopping(rounds=20)])
        model.fit(
            X_train, y_train.ravel(),
            eval_set=eval_set,
            verbose=False
        )
    else:
        model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'r2_score': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        importance = np.zeros(X.shape[1])
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return metrics, feature_importance, model, scaler, X_test, y_test, y_pred




def clean_model_params(model):
    """Clean model parameters for JSON serialization"""
    params = model.get_params()
    # Remove callbacks and other non-serializable items
    if 'callbacks' in params:
        del params['callbacks']
    return params





def create_and_save_pipeline(model, scaler, X, feature_columns, base_path='models') -> ModelPipeline:
    """Create and save pipeline from model with metadata"""
    try:
        pipeline = ModelPipeline()
        
        # Identify base features (features without derived indicators)
        base_features = [col for col in feature_columns 
                        if not any(x in col for x in ['_lag_', 'ma_', 'momentum_', 'price_rel_'])]
        
        # Create additional metadata with cleaned parameters
        additional_metadata = {
            'training_shape': X.shape,
            'model_params': clean_model_params(model),
            'base_features': base_features,
            'feature_types': {
                'base': base_features,
                'technical': [col for col in feature_columns if any(x in col for x in ['ma_', 'momentum_', 'price_rel_'])],
                'lagged': [col for col in feature_columns if '_lag_' in col],
                'time': ['hour', 'day_of_week', 'day_of_month', 'month']
            }
        }
        
        # Save pipeline
        pipeline.save_pipeline(
            model=model,
            feature_scaler=scaler,
            target_scaler=scaler,  # Using same scaler for both as we only scaled features
            feature_columns=feature_columns,
            base_path=base_path,
            additional_metadata=additional_metadata
        )
        
        return pipeline
        
    except Exception as e:
        print(f"Error in create_and_save_pipeline: {str(e)}")  # Debug log
        raise Exception(f"Error in pipeline creation: {str(e)}")

def save_trained_model(model, scaler, features, X, target_col) -> Tuple[bool, str]:
    """Save trained model using pipeline approach"""
    try:
        # Create log container in UI
        log_container = st.empty()
        log_container.write("Starting model save process...")
        
        # Create timestamp for model directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f'ml_model_{timestamp}'
        model_path = os.path.join('models', model_name)
        
        # Create directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        log_container.write(f"Created directory: {model_path}")
        
        # Create and save pipeline
        pipeline = create_and_save_pipeline(
            model=model,
            scaler=scaler,
            X=X,
            feature_columns=features,
            base_path=model_path
        )
        
        log_container.write("✅ Model saved successfully!")
        return True, f"Model saved successfully to {model_path}"
        
    except Exception as e:
        error_msg = f"Error saving model: {str(e)}"
        print(f"Error details: {error_msg}")  # Terminal log
        log_container.write(f"❌ {error_msg}")
        return False, error_msg





def plot_results(X_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, target_col: str):
    """Plot regression results with enhanced visualizations"""
    # Create figure for actual vs predicted comparison
    fig = go.Figure()
    
    # Time series plot
    fig.add_trace(go.Scatter(
        y=y_test,
        name='Actual',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        y=y_pred,
        name='Predicted',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title=f'Actual vs Predicted {target_col} Over Time',
        xaxis_title='Sample Index',
        yaxis_title='Value',
        showlegend=True
    )
    st.plotly_chart(fig)
    
    # Scatter plot
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        marker=dict(color='blue', size=8, opacity=0.6),
        name='Data Points'
    ))
    
    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction'
    ))
    
    fig_scatter.update_layout(
        title='Actual vs Predicted Scatter Plot',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        showlegend=True
    )
    st.plotly_chart(fig_scatter)
    
    # Additional statistics
    st.subheader("Additional Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Statistical Summary:")
        stats_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        st.write(stats_df.describe())
    
    with col2:
        st.write("Correlation Analysis:")
        correlation = np.corrcoef(y_test, y_pred)[0, 1]
        st.metric("Correlation Coefficient", f"{correlation:.3f}")



def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_run' not in st.session_state:
        st.session_state.analysis_run = False
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = None
    if 'target_col' not in st.session_state:
        st.session_state.target_col = None

def run_analysis_clicked():
    st.session_state.analysis_run = True

def sklearn_page():
    """Enhanced ML analysis page implementation with XGBoost"""
    st.title("ML Analysis with Scikit-learn and XGBoost")
    
    # Initialize session state
    initialize_session_state()
    
    # File upload
    file_path = st.text_input("Enter the path to your CSV file:", value="")
    
    if not file_path or not os.path.exists(file_path):
        st.error("Please provide a valid file path")
        return
    
    try:
        # Load and display data
        df = pd.read_csv(file_path)
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        
        # Feature selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Available Features")
            available_features = list(df.columns)
            selected_features = st.multiselect("Select Features", available_features)
            if selected_features:
                st.session_state.selected_features = selected_features
        
        with col2:
            st.subheader("Algorithm Configuration")
            target_col = st.selectbox(
                "Select Target Variable",
                selected_features if selected_features else []
            )
            if target_col:
                st.session_state.target_col = target_col
            
            algorithm = st.selectbox(
                "Select Regression Algorithm",
                ["xgboost", "linear", "ridge", "lasso", "random_forest", "gradient_boosting"]
            )
            
            # Get algorithm-specific parameters
            params = get_algorithm_params(algorithm)
        
        if selected_features and target_col:
            if st.button("Run Analysis", key='run_analysis', on_click=run_analysis_clicked):
                pass  # The actual analysis will be triggered by the state change
            
            # Check if analysis should be run
            if st.session_state.analysis_run:
                with st.spinner("Processing data..."):
                    try:
                        # Run regression
                        metrics, feature_importance, model, scaler, X_test, y_test, y_pred = run_regression(
                            df[selected_features], target_col, algorithm, params
                        )
                        
                        # Store results in session state
                        st.session_state.analysis_data = {
                            'model': model,
                            'scaler': scaler,
                            'X_scaled': X_test,  # Store the scaled features
                            'metrics': metrics,
                            'feature_importance': feature_importance,
                            'y_test': y_test,
                            'y_pred': y_pred
                        }
                        
                        # Display results
                        st.subheader("Regression Results")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("R² Score", f"{metrics['r2_score']:.3f}")
                        col2.metric("MSE", f"{metrics['mse']:.3f}")
                        col3.metric("RMSE", f"{metrics['rmse']:.3f}")
                        
                        # Display feature importance
                        st.subheader("Feature Importance")
                        fig = px.bar(
                            feature_importance,
                            x='importance',
                            y='feature',
                            orientation='h',
                            title='Feature Importance'
                        )
                        st.plotly_chart(fig)
                        
                        # Plot results
                        plot_results(X_test, y_test, y_pred, target_col)
                        
                        # Add save model button
                        save_col1, save_col2 = st.columns([1, 2])
                        with save_col1:
                            if st.button("Save Model", key='save_model'):
                                st.write("Saving model...")
                                success, message = save_trained_model(
                                    model=model,
                                    scaler=scaler,
                                    features=selected_features,
                                    X=X_test,
                                    target_col=target_col
                                )
                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)
                    
                    except Exception as e:
                        st.error(f"Error in analysis: {str(e)}")
                        print(f"Error details: {str(e)}")  # Terminal log
        
        else:
            st.warning("Please select features and a target variable to proceed")
    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")


if __name__ == "__main__":
    sklearn_page()