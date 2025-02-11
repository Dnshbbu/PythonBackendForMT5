import streamlit as st
from train_pycaret_models_page import train_pycaret_models_page

def automatic_ml_page():
    """Main page for Automatic ML using PyCaret"""
    
    st.markdown("""
        <h2 style='text-align: center;'>🤖 Automatic ML - PyCaret</h2>
        <p style='text-align: center; color: #666;'>
            Automated machine learning tools for strategy optimization and prediction
        </p>
        <hr>
    """, unsafe_allow_html=True)
    
    # Create tabs for different functionalities
    tabs = st.tabs([
        "🎯 Train Models",
        "🔮 Run Predictions",
        "📊 Model Analysis",
        "⚙️ Model Management"
    ])
    
    # Train Models Tab
    with tabs[0]:
        train_pycaret_models_page()
    
    # Run Predictions Tab
    with tabs[1]:
        st.markdown("""
            <h3 style='text-align: center;'>🔮 Run Predictions</h3>
            <p style='text-align: center; color: #666;'>
                Coming soon! This tab will allow you to make predictions using trained PyCaret models.
            </p>
        """, unsafe_allow_html=True)
    
    # Model Analysis Tab
    with tabs[2]:
        st.markdown("""
            <h3 style='text-align: center;'>📊 Model Analysis</h3>
            <p style='text-align: center; color: #666;'>
                Coming soon! This tab will provide detailed analysis of trained models.
            </p>
        """, unsafe_allow_html=True)
    
    # Model Management Tab
    with tabs[3]:
        st.markdown("""
            <h3 style='text-align: center;'>⚙️ Model Management</h3>
            <p style='text-align: center; color: #666;'>
                Coming soon! This tab will allow you to manage your trained PyCaret models.
            </p>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    automatic_ml_page() 