import streamlit as st
import pandas as pd
from typing import List, Dict
import os
from pycaret_model_predictor import PyCaretModelPredictor
from db_info import get_table_names
import logging

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_available_models(models_dir: str) -> List[str]:
    """Get list of available PyCaret models"""
    return [d for d in os.listdir(models_dir) 
            if os.path.isdir(os.path.join(models_dir, d))
            and d.startswith("pycaret_model_")]

def pycaret_predictions_page():
    """Streamlit page for making predictions with PyCaret models"""
    st.title("PyCaret Model Predictions")
    st.write("""
    This page allows you to make predictions using trained PyCaret models.
    You can select a specific model or use the most recently trained one.
    """)
    
    # Initialize predictor
    db_path = "trading_data.db"
    models_dir = "models"
    predictor = PyCaretModelPredictor(db_path, models_dir)
    
    # Get available models and tables
    available_models = get_available_models(models_dir)
    tables = get_table_names(db_path)
    
    # Model selection
    st.subheader("Model Selection")
    use_latest = st.checkbox("Use latest model", value=True)
    
    if not use_latest:
        if not available_models:
            st.error("No PyCaret models found")
            return
            
        selected_model = st.selectbox(
            "Select model",
            available_models,
            help="Choose a trained PyCaret model"
        )
    
    # Table selection
    st.subheader("Data Selection")
    selected_table = st.selectbox(
        "Select table for prediction",
        tables,
        help="Choose the table to get data from"
    )
    
    n_rows = st.number_input(
        "Number of recent rows to use",
        min_value=1,
        value=100,
        help="Number of most recent data rows to use for prediction"
    )
    
    # Make prediction button
    if st.button("Make Prediction"):
        with st.spinner("Making prediction..."):
            try:
                # Load model
                if use_latest:
                    predictor.load_latest_model()
                    st.info(f"Using latest model: {predictor.current_model_name}")
                else:
                    predictor.load_model_by_name(selected_model)
                    st.info(f"Using model: {selected_model}")
                
                # Make prediction
                result = predictor.make_predictions(selected_table, n_rows)
                
                # Display results
                st.subheader("Prediction Results")
                
                # Main prediction
                st.metric(
                    "Predicted Value",
                    f"{result['prediction']:.4f}"
                )
                
                # Metadata
                st.write(f"Model: {result['model_name']}")
                st.write(f"Timestamp: {result['timestamp']}")
                
                # Feature importance
                if result['feature_importance']:
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame(
                        list(result['feature_importance'].items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=False)
                    
                    st.bar_chart(
                        importance_df.set_index('Feature')['Importance']
                    )
                
                # Explanation
                st.subheader("Prediction Explanation")
                explanation = predictor.get_prediction_explanation(result)
                st.write(explanation)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    setup_logging()
    pycaret_predictions_page() 