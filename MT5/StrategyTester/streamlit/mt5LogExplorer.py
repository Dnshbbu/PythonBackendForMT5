# mt5LogExplorer.py
import streamlit as st
from log_explorer_page import log_explorer
from scripts_page import scripts
from zmq_server_page import zmq_server
from ml_analysis_page import sklearn_page

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
        "ZMQ Server": zmq_server,
        "ML Analysis": sklearn_page
    }
    
    # Add the navigation to the sidebar
    page = st.sidebar.selectbox("Navigation", list(pages.keys()))
    
    # Call the selected page function
    pages[page]()

if __name__ == "__main__":
    main()