# mt5LogExplorer.py
import streamlit as st
from log_explorer_page import log_explorer
from scripts_page import scripts
from ml_analysis_page import sklearn_page
from server_control_page import server_control_page


import pandas as pd
from model_manager import ModelManager


def prediction_page():
    st.title("Model Predictions")
    
    model_manager = ModelManager()
    models = model_manager.list_models()
    
    if not models:
        st.warning("No saved models found")
        return
    
    selected_model = st.selectbox(
        "Select Model",
        options=models,
        format_func=lambda x: f"{x.name} ({x.algorithm} - {x.creation_date})"
    )
    
    uploaded_file = st.file_uploader("Upload data for predictions", type=['csv'])
    if uploaded_file and selected_model:
        try:
            data = pd.read_csv(uploaded_file)
            predictions = model_manager.predict(selected_model.name, data)
            
            results = data.copy()
            results[f'Predicted_{selected_model.target}'] = predictions
            
            st.write("Predictions:")
            st.dataframe(results)
            
            csv = results.to_csv(index=False)
            st.download_button(
                "Download Predictions",
                csv,
                "predictions.csv",
                "text/csv"
            )
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")



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