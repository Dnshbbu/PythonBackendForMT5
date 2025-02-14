import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
from time_series_predictor import (
    load_model,
    prepare_data_for_prediction,
    make_predictions,
    get_model_info
)

def initialize_ts_pred_session_state():
    """Initialize session state variables for time series prediction page"""
    if 'ts_pred_model_path' not in st.session_state:
        st.session_state['ts_pred_model_path'] = None
    if 'ts_pred_model_info' not in st.session_state:
        st.session_state['ts_pred_model_info'] = None
    if 'ts_pred_data' not in st.session_state:
        st.session_state['ts_pred_data'] = None

def display_model_info(model_info: dict):
    """Display model information in a formatted way"""
    st.markdown("### 📊 Model Information")
    
    # Create columns for different types of information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Model Details")
        st.write(f"**Model Type:** {model_info.get('model_type', 'N/A')}")
        st.write(f"**Training Date:** {model_info.get('training_date', 'N/A')}")
        
        # Display model parameters if available
        if 'model_params' in model_info:
            st.markdown("#### ⚙️ Model Parameters")
            for param, value in model_info['model_params'].items():
                st.write(f"**{param}:** {value}")
    
    with col2:
        st.markdown("#### 📈 Features")
        if 'features' in model_info:
            st.write("**Required Features:**")
            for feature in model_info['features']:
                st.write(f"- {feature}")
        
        if 'target_column' in model_info:
            st.write(f"**Target Column:** {model_info['target_column']}")

def predict_time_series_page():
    """Streamlit interface for time series predictions"""
    # st.markdown("""
    #     <h2 style='text-align: center;'>📈 Time Series Predictions</h2>
    #     <p style='text-align: center; color: #666;'>
    #         Make predictions using trained time series models
    #     </p>
    #     <hr>
    # """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_ts_pred_session_state()
    
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, 'models', 'time_series')
    
    # Model Selection Section
    st.markdown("### 🤖 Select Model")
    
    # Get available models
    available_models = []
    if os.path.exists(models_dir):
        for model_dir in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_dir)
            if os.path.isdir(model_path):
                available_models.append(model_dir)
    
    if not available_models:
        st.warning("No trained models found. Please train a model first.")
        return
    
    # Model selection dropdown
    selected_model = st.selectbox(
        "Select a trained model",
        options=available_models,
        help="Choose a trained time series model for making predictions"
    )
    
    if selected_model:
        model_path = os.path.join(models_dir, selected_model)
        
        # Load model info if not already loaded or if model changed
        if (st.session_state['ts_pred_model_path'] != model_path or 
            st.session_state['ts_pred_model_info'] is None):
            try:
                model_info = get_model_info(model_path)
                st.session_state['ts_pred_model_info'] = model_info
                st.session_state['ts_pred_model_path'] = model_path
            except Exception as e:
                st.error(f"Error loading model information: {str(e)}")
                return
        
        # Display model information
        display_model_info(st.session_state['ts_pred_model_info'])
        
        # Data Input Section
        st.markdown("### 📊 Input Data")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your data file",
            type=['csv'],
            help="Upload a CSV file containing the required features"
        )
        
        if uploaded_file:
            try:
                # Read the data
                data = pd.read_csv(uploaded_file)
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(data.head())
                
                # Verify required features
                required_features = st.session_state['ts_pred_model_info'].get('features', [])
                missing_features = [f for f in required_features if f not in data.columns]
                
                if missing_features:
                    st.error(f"Missing required features: {', '.join(missing_features)}")
                    return
                
                # Make predictions
                if st.button("🚀 Generate Predictions", type="primary"):
                    with st.spinner("Generating predictions..."):
                        try:
                            # Load the model
                            model = load_model(model_path)
                            
                            # Prepare data for prediction
                            pred_data = prepare_data_for_prediction(
                                data,
                                st.session_state['ts_pred_model_info']
                            )
                            
                            # Make predictions
                            predictions = make_predictions(model, pred_data)
                            
                            # Add predictions to results
                            results = data.copy()
                            target_col = st.session_state['ts_pred_model_info'].get('target_column', 'target')
                            results[f'Predicted_{target_col}'] = predictions
                            
                            # Display results
                            st.subheader("Predictions")
                            st.dataframe(results)
                            
                            # Download option
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions",
                                data=csv,
                                file_name=f"ts_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                            # Display prediction statistics
                            st.subheader("Prediction Statistics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Mean", f"{predictions.mean():.4f}")
                            with col2:
                                st.metric("Min", f"{predictions.min():.4f}")
                            with col3:
                                st.metric("Max", f"{predictions.max():.4f}")
                            
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
                            logging.error(f"Prediction error: {str(e)}", exc_info=True)
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                logging.error(f"File reading error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    predict_time_series_page() 