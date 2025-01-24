"""
ml_analysis_page.py - Machine Learning Analysis Page Implementation
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional, Any

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, accuracy_score, 
    precision_score, recall_score, f1_score
)

# Model imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

import plotly.express as px
import plotly.graph_objects as go
from model_manager import ModelManager, create_model_info

def initialize_session_state():
    """Initialize session state variables"""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'features' not in st.session_state:
        st.session_state.features = None
    if 'target' not in st.session_state:
        st.session_state.target = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None

def run_regression(
    data: pd.DataFrame,
    target_col: str,
    algorithm: str,
    params: Dict
) -> Tuple[Dict[str, float], pd.DataFrame, Any, Any, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run regression analysis with specified algorithm
    
    Args:
        data: Input DataFrame
        target_col: Target variable name
        algorithm: Algorithm name
        params: Algorithm parameters
        
    Returns:
        Tuple containing metrics, feature importance, model, scaler, and predictions
    """
    # Prepare data
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.2, random_state=42
    )
    
    # Initialize model
    if algorithm == "linear":
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
    
    # Train model and predict
    model.fit(X_train, y_train)
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

def plot_results(X_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, target_col: str):
    """Plot regression results"""
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=np.arange(len(y_test)),
        y=y_test,
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))
    
    # Predicted values
    fig.add_trace(go.Scatter(
        x=np.arange(len(y_pred)),
        y=y_pred,
        mode='lines',
        name='Predicted',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title=f'Actual vs Predicted {target_col}',
        xaxis_title='Sample Index',
        yaxis_title='Value',
        showlegend=True
    )
    
    st.plotly_chart(fig)

def save_trained_model() -> Tuple[bool, str]:
    """
    Save the trained model
    
    Returns:
        Tuple[bool, str]: Success status and message
    """
    try:
        if not st.session_state.model:
            return False, "No trained model found. Please run analysis first."
        
        model_manager = ModelManager()
        
        # Create model info
        model_name = f"{st.session_state.algorithm}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_info = create_model_info(
            name=model_name,
            model_type="regression",
            algorithm=st.session_state.algorithm,
            features=st.session_state.features,
            target=st.session_state.target,
            metrics=st.session_state.metrics
        )
        
        # Save model
        model_dir = model_manager.save_model(
            st.session_state.model,
            st.session_state.scaler,
            model_info
        )
        
        return True, f"Model saved successfully to {model_dir}"
        
    except Exception as e:
        return False, f"Error saving model: {str(e)}"

def sklearn_page():
    """Main ML analysis page implementation"""
    st.title("ML Analysis with Scikit-learn")
    
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
        
        with col2:
            st.subheader("Algorithm Configuration")
            target_col = st.selectbox(
                "Select Target Variable",
                selected_features
            ) if selected_features else None
            
            algorithm = st.selectbox(
                "Select Regression Algorithm",
                ["linear", "ridge", "lasso", "random_forest", "gradient_boosting"]
            )
            
            # Algorithm parameters
            params = {}
            if algorithm in ["ridge", "lasso"]:
                params['alpha'] = st.slider(
                    "Alpha",
                    0.0, 1.0, 0.1
                )
            elif algorithm in ["random_forest", "gradient_boosting"]:
                params['n_estimators'] = st.slider(
                    "Number of Estimators",
                    10, 500, 100
                )
        
        if selected_features and target_col:
            if st.button("Run Analysis", type="primary"):
                with st.spinner("Processing data..."):
                    try:
                        # Run regression
                        metrics, feature_importance, model, scaler, X_test, y_test, y_pred = run_regression(
                            df[selected_features], target_col, algorithm, params
                        )
                        
                        # Store in session state
                        st.session_state.model = model
                        st.session_state.scaler = scaler
                        st.session_state.features = selected_features
                        st.session_state.target = target_col
                        st.session_state.metrics = metrics
                        st.session_state.algorithm = algorithm
                        
                        # Display results
                        st.subheader("Regression Results")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("R² Score", f"{metrics['r2_score']:.3f}")
                        col2.metric("MSE", f"{metrics['mse']:.3f}")
                        col3.metric("RMSE", f"{metrics['rmse']:.3f}")
                        
                        st.subheader("Feature Importance")
                        st.dataframe(feature_importance)
                        
                        # Plot results
                        plot_results(X_test, y_test, y_pred, target_col)
                        
                        # # Add save model button
                        # if st.button("Save Model"):
                        #     success, message = save_trained_model()
                        #     if success:
                        #         st.success(message)
                        #     else:
                        #         st.error(message)


                        # Add save model button
                        st.write("Debug - About to show save button")
                        if st.button("Save Model"):
                            st.write("Debug - Save button clicked")
                            success, message = save_trained_model()
                            st.write(f"Debug - Save result: {success}")
                            if success:
                                st.success(message)
                            else:
                                st.error(message)        
                    
                    except Exception as e:
                        st.error(f"Error in analysis: {str(e)}")
        
        else:
            st.warning("Please select features and a target variable to proceed")
    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

if __name__ == "__main__":
    sklearn_page()