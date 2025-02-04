# mt5LogExplorer.py
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from log_explorer_page import log_explorer
from scripts_page import scripts
# from ml_analysis_page import sklearn_page
from server_control_page import server_control_page
# from usingXgboost import ModelPipeline 
# from model_pipeline import ModelPipeline, create_pipeline_from_analyzer
from prediction_monitoring_page import prediction_monitoring_page
from historical_predictions_page import historical_predictions_page
# from realtime_monitoring_page import realtime_monitoring
from model_comparison_page import model_comparison_page



# def prediction_page():
#     """Model Predictions page implementation with improved feature validation"""
#     st.title("Model Predictions")
    
#     # Initialize ModelPipeline
#     pipeline = ModelPipeline()
    
#     # Get list of available models
#     models_dir = "models"
#     if not os.path.exists(models_dir):
#         st.error("No models directory found. Please train a model first.")
#         return
    
#     # Scan for model files
#     model_files = []
#     for item in os.listdir(models_dir):
#         item_path = os.path.join(models_dir, item)
#         if os.path.isfile(item_path) and item.endswith('model_pipeline.joblib'):
#             model_files.append(item_path)
#         elif os.path.isdir(item_path):
#             subdir_model = os.path.join(item_path, 'model_pipeline.joblib')
#             if os.path.isfile(subdir_model):
#                 model_files.append(subdir_model)
    
#     if not model_files:
#         st.error("No trained models found. Please train a model first.")
#         return
    
#     # Model selection
#     selected_model = st.selectbox(
#         "Select Model",
#         options=model_files,
#         format_func=lambda x: os.path.dirname(x) if os.path.dirname(x) else os.path.basename(x)
#     )
    
#     # Load the selected model
#     try:
#         model_path = os.path.dirname(selected_model) if os.path.dirname(selected_model) else models_dir
#         pipeline.load_pipeline(model_path)
        
#         # Display model metadata
#         with st.expander("Model Information", expanded=True):
#             metadata = pipeline.get_metadata()
            
#             if 'feature_types' in metadata:
#                 st.write("### Required Input Data")
                
#                 # Show base features that user must provide
#                 st.write("#### Base Features (Must be in your input data):")
#                 st.write(metadata['feature_types']['base'])
                
#                 # Show derived features that will be automatically generated
#                 st.write("#### Features that will be automatically generated:")
#                 st.write("1. Time features (requires Date and Time columns):")
#                 st.write(metadata['feature_types']['time'])
#                 st.write("2. Technical features (requires Price column):")
#                 st.write(metadata['feature_types']['technical'])
#                 st.write("3. Lagged features (automatically created from base features):")
#                 st.write(metadata['feature_types']['lagged'])
    
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         return
    
#     # File upload section
#     uploaded_file = st.file_uploader(
#         "Upload data for predictions", 
#         type=['csv'],
#         help="Upload a CSV file containing the required base features"
#     )
    
#     if uploaded_file:
#         try:
#             # Read the data
#             data = pd.read_csv(uploaded_file)
            
#             # Display data preview
#             st.subheader("Data Preview")
#             st.dataframe(data.head())
            
#             # Verify required columns for feature generation
#             required_columns = []
#             missing_required = []
            
#             # Check for datetime columns
#             if ('Date' not in data.columns or 'Time' not in data.columns) and 'DateTime' not in data.columns:
#                 missing_required.append("Date and Time columns (or DateTime column)")
            
#             # Check for Price column if technical features are needed
#             if any('price_rel_' in f or 'ma_' in f for f in pipeline.feature_columns):
#                 if 'Price' not in data.columns:
#                     missing_required.append("Price column")
            
#             # Check base features
#             base_features = metadata.get('feature_types', {}).get('base', [])
#             missing_base = [f for f in base_features if f not in data.columns]
            
#             if missing_required or missing_base:
#                 if missing_required:
#                     st.error(f"Missing required columns: {', '.join(missing_required)}")
#                 if missing_base:
#                     st.error(f"Missing base features: {missing_base}")
#                 return
            
#             # Make predictions
#             with st.spinner("Generating predictions..."):
#                 predictions = pipeline.predict(data)
            
#             # Add predictions to results
#             results = data.copy()
#             results['Predicted_Value'] = predictions
            
#             # Display results
#             st.subheader("Predictions")
#             st.dataframe(results)
            
#             # Download option
#             csv = results.to_csv(index=False)
#             st.download_button(
#                 label="Download Predictions",
#                 data=csv,
#                 file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                 mime="text/csv"
#             )
            
#             # Statistics
#             st.subheader("Prediction Statistics")
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.metric("Mean Prediction", f"{predictions.mean():.2f}")
#             with col2:
#                 st.metric("Min Prediction", f"{predictions.min():.2f}")
#             with col3:
#                 st.metric("Max Prediction", f"{predictions.max():.2f}")
            
#         except Exception as e:
#             st.error("Error during prediction:")
#             with st.expander("Error Details"):
#                 st.code(str(e))



def main():
    st.set_page_config(
        page_title="MT5 Analysis Tools",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Page navigation
    pages = {
        "Log Explorer": log_explorer,
        "Scripts": scripts,
        # "ZMQ Server": zmq_server,
        # "ML: Analysis": sklearn_page,
        # "ML: Predictions": prediction_page,
        "ZMQ Server Control": server_control_page,
        "Real-Time Prediction Monitor": prediction_monitoring_page,
        "Historical Predictions Analysis": historical_predictions_page,
        "Model Comparison": model_comparison_page
    }
    
    # Add the navigation to the sidebar
    page = st.sidebar.selectbox("Navigation", list(pages.keys()))
    
    # Call the selected page function
    pages[page]()

if __name__ == "__main__":
    main()