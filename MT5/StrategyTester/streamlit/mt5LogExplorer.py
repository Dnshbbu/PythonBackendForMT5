# mt5LogExplorer.py
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from log_explorer_page import log_explorer
from scripts_page import scripts
from ml_analysis_page import sklearn_page
from server_control_page import server_control_page
# from usingXgboost import ModelPipeline 
from model_pipeline import ModelPipeline, create_pipeline_from_analyzer



import pandas as pd
from model_manager import ModelManager


# def prediction_page():
#     st.title("Model Predictions")
    
#     model_manager = ModelManager()
#     models = model_manager.list_models()
    
#     if not models:
#         st.warning("No saved models found")
#         return
    
#     selected_model = st.selectbox(
#         "Select Model",
#         options=models,
#         format_func=lambda x: f"{x.name} ({x.algorithm} - {x.creation_date})"
#     )
    
#     uploaded_file = st.file_uploader("Upload data for predictions", type=['csv'])
#     if uploaded_file and selected_model:
#         try:
#             data = pd.read_csv(uploaded_file)
#             predictions = model_manager.predict(selected_model.name, data)
            
#             results = data.copy()
#             results[f'Predicted_{selected_model.target}'] = predictions
            
#             st.write("Predictions:")
#             st.dataframe(results)
            
#             csv = results.to_csv(index=False)
#             st.download_button(
#                 "Download Predictions",
#                 csv,
#                 "predictions.csv",
#                 "text/csv"
#             )
#         except Exception as e:
#             st.error(f"Error making predictions: {str(e)}")


# def prediction_page():
#     """Enhanced prediction page with more functionality"""
#     st.title("Model Predictions")
    
#     # Create a dictionary to store the required features and their default values
#     required_features = {
#         'EntryScore_SR': 0.0,
#         'EntryScore_Pullback': 0.0,
#         'EntryScore_EMA': 0.0,
#         'EntryScore_AVWAP': 0.0,
#         'Factors_srScore': 0.0,
#         'Factors_maScore': 0.0,
#         'Factors_rsiScore': 0.0,
#         'Factors_macdScore': 0.0,
#         'Factors_stochScore': 0.0,
#         'Factors_bbScore': 0.0,
#         'Factors_atrScore': 0.0,
#         'Factors_sarScore': 0.0,
#         'Factors_ichimokuScore': 0.0,
#         'Factors_adxScore': 0.0,
#         'Factors_volumeScore': 0.0,
#         'Factors_mfiScore': 0.0,
#         'Factors_priceMAScore': 0.0,
#         'Factors_emaScore': 0.0,
#         'Factors_emaCrossScore': 0.0,
#         'Factors_cciScore': 0.0
#     }
    
#     # Create tabs for different input methods
#     tab1, tab2 = st.tabs(["Manual Input", "CSV Upload"])
    
#     with tab1:
#         st.header("Enter Feature Values")
        
#         # Create columns for better layout
#         col1, col2 = st.columns(2)
#         input_values = {}
        
#         # Distribute features between columns
#         features_list = list(required_features.keys())
#         half = len(features_list) // 2
        
#         # First column
#         with col1:
#             for feature in features_list[:half]:
#                 input_values[feature] = st.number_input(
#                     feature,
#                     value=float(required_features[feature]),
#                     format="%.4f"
#                 )
        
#         # Second column
#         with col2:
#             for feature in features_list[half:]:
#                 input_values[feature] = st.number_input(
#                     feature,
#                     value=float(required_features[feature]),
#                     format="%.4f"
#                 )
        
#         if st.button("Predict", key="manual_predict"):
#             try:
#                 # Create DataFrame from input values
#                 input_df = pd.DataFrame([input_values])
                
#                 # Load model pipeline
#                 pipeline = ModelPipeline()
#                 pipeline.load_pipeline()
                
#                 # Make prediction
#                 prediction = pipeline.predict(input_df)
                
#                 # Display prediction with confidence styling
#                 st.subheader("Prediction Results")
#                 st.metric(
#                     label="Predicted Price",
#                     value=f"{prediction[0]:.4f}"
#                 )
                
#             except Exception as e:
#                 st.error(f"Error making prediction: {str(e)}")
    
#     with tab2:
#         st.header("Upload CSV File")
#         uploaded_file = st.file_uploader("Upload data for predictions", type=['csv'])
        
#         if uploaded_file:
#             try:
#                 data = pd.read_csv(uploaded_file)
                
#                 # Show data preview
#                 st.subheader("Data Preview")
#                 st.dataframe(data.head())
                
#                 # Check for missing required features
#                 missing_features = set(required_features.keys()) - set(data.columns)
#                 if missing_features:
#                     st.error(f"Missing required features in CSV: {missing_features}")
#                 else:
#                     if st.button("Predict", key="csv_predict"):
#                         # Load model pipeline
#                         pipeline = ModelPipeline()
#                         pipeline.load_pipeline()
                        
#                         # Make predictions
#                         predictions = pipeline.predict(data)
                        
#                         # Add predictions to data
#                         results = data.copy()
#                         results['Predicted_Price'] = predictions
                        
#                         # Display results
#                         st.subheader("Prediction Results")
#                         st.dataframe(results)
                        
#                         # Create download button for results
#                         csv = results.to_csv(index=False)
#                         st.download_button(
#                             "Download Predictions",
#                             csv,
#                             "predictions.csv",
#                             "text/csv"
#                         )
                        
#                         # Plot actual vs predicted if 'Price' column exists
#                         if 'Price' in data.columns:
#                             st.subheader("Actual vs Predicted Plot")
#                             fig, ax = plt.subplots(figsize=(10, 6))
#                             ax.scatter(data['Price'], predictions, alpha=0.5)
#                             ax.plot([data['Price'].min(), data['Price'].max()],
#                                   [data['Price'].min(), data['Price'].max()],
#                                   'r--', alpha=0.8)
#                             ax.set_xlabel('Actual Price')
#                             ax.set_ylabel('Predicted Price')
#                             ax.set_title('Actual vs Predicted Prices')
#                             st.pyplot(fig)
                            
#                             # Calculate and display metrics
#                             mse = mean_squared_error(data['Price'], predictions)
#                             mae = mean_absolute_error(data['Price'], predictions)
#                             r2 = r2_score(data['Price'], predictions)
                            
#                             col1, col2, col3 = st.columns(3)
#                             col1.metric("MSE", f"{mse:.6f}")
#                             col2.metric("MAE", f"{mae:.6f}")
#                             col3.metric("R²", f"{r2:.6f}")
                
#             except Exception as e:
#                 st.error(f"Error processing file: {str(e)}")

def prediction_page():
    """Model Predictions page implementation"""
    st.title("Model Predictions")
    
    # Initialize ModelPipeline
    pipeline = ModelPipeline()
    
    # Get list of available models
    models_dir = "models"
    if not os.path.exists(models_dir):
        st.error("No models directory found. Please train a model first.")
        return
    
    # Scan for model files (not directories)
    model_files = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        # Check if it's a file and ends with .joblib
        if os.path.isfile(item_path) and item.endswith('model_pipeline.joblib'):
            model_files.append(item_path)
        # Also check subdirectories
        elif os.path.isdir(item_path):
            subdir_model = os.path.join(item_path, 'model_pipeline.joblib')
            if os.path.isfile(subdir_model):
                model_files.append(subdir_model)
    
    if not model_files:
        st.error("No trained models found. Please train a model first.")
        return
    
    # Model selection
    selected_model = st.selectbox(
        "Select Model",
        options=model_files,
        format_func=lambda x: os.path.dirname(x) if os.path.dirname(x) else os.path.basename(x),
        help="Choose a trained model to use for predictions"
    )
    
    # Load the selected model
    try:
        model_path = os.path.dirname(selected_model) if os.path.dirname(selected_model) else models_dir
        pipeline.load_pipeline(model_path)
        
        # Display model metadata in an expander
        with st.expander("Model Information", expanded=False):
            metadata = pipeline.get_metadata()
            st.json(metadata)
            
            # Display feature columns if available
            feature_cols = pipeline.get_feature_columns()
            if feature_cols:
                st.write("Required Features:")
                st.write(feature_cols)
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.write("Error details:")
        st.code(str(e))
        return
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload data for predictions", 
        type=['csv'],
        help="Upload a CSV file containing the required features for prediction"
    )
    
    if uploaded_file:
        try:
            # Read the data
            data = pd.read_csv(uploaded_file)
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            # Check for required features
            feature_cols = pipeline.get_feature_columns()
            if feature_cols:
                missing_features = set(feature_cols) - set(data.columns)
                if missing_features:
                    st.error(f"Missing required features in uploaded file: {missing_features}")
                    return
            
            # Make predictions
            with st.spinner("Generating predictions..."):
                predictions = pipeline.predict(data)
            
            # Add predictions to the results
            results = data.copy()
            results['Predicted_Value'] = predictions
            
            # Display results
            st.subheader("Predictions")
            st.dataframe(results)
            
            # Provide download option
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Display basic statistics
            st.subheader("Prediction Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Prediction", f"{predictions.mean():.2f}")
            with col2:
                st.metric("Min Prediction", f"{predictions.min():.2f}")
            with col3:
                st.metric("Max Prediction", f"{predictions.max():.2f}")
            
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            st.write("Error details:")
            st.code(str(e))
            
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
        "ML: Analysis": sklearn_page,
        "ML: Predictions": prediction_page,
        "Server Control": server_control_page
    }
    
    # Add the navigation to the sidebar
    page = st.sidebar.selectbox("Navigation", list(pages.keys()))
    
    # Call the selected page function
    pages[page]()

if __name__ == "__main__":
    main()