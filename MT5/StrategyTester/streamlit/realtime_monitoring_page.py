"""
realtime_monitoring_page.py - Streamlit page for real-time ML monitoring
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time
import os

def create_candlestick_chart(df):
    """Create candlestick chart with predictions"""
    fig = go.Figure()
    
    # Add actual price line
    fig.add_trace(go.Scatter(
        x=df['Timestamp'],
        y=df['Actual_Price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='blue')
    ))
    
    # Add predicted price line
    fig.add_trace(go.Scatter(
        x=df['Timestamp'],
        y=df['Predicted_Price'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='Price vs Predictions',
        xaxis_title='Time',
        yaxis_title='Price',
        height=600
    )
    
    return fig

def create_metrics_chart(df):
    """Create metrics chart showing prediction accuracy"""
    df['Prediction_Error'] = abs(df['Actual_Price'] - df['Predicted_Price'])
    df['Error_Percentage'] = (df['Prediction_Error'] / df['Actual_Price']) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Timestamp'],
        y=df['Error_Percentage'],
        mode='lines',
        name='Prediction Error %',
        line=dict(color='orange')
    ))
    
    fig.update_layout(
        title='Prediction Error Percentage',
        xaxis_title='Time',
        yaxis_title='Error %',
        height=400
    )
    
    return fig

def realtime_monitoring():
    """Real-time monitoring page implementation"""
    st.title("Real-Time ML Monitoring")
    
    # Initialize session state
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    # Add refresh button and auto-refresh checkbox
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Refresh"):
            st.session_state.last_refresh = datetime.now()
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)
    
    # Auto-refresh logic
    if auto_refresh:
        time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
        if time_since_refresh >= 5:
            st.session_state.last_refresh = datetime.now()
            st.rerun()  # Updated from experimental_rerun to rerun
    
    try:
        # Get predictions directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        predictions_dir = os.path.join(base_dir, 'predictions')
        
        if not os.path.exists(predictions_dir):
            st.warning("Predictions directory not found. Waiting for data...")
            return
        
        # Get latest predictions file
        prediction_files = [f for f in os.listdir(predictions_dir) if f.startswith('predictions_')]
        if not prediction_files:
            st.warning("No prediction data available yet. Waiting for predictions...")
            return
        
        latest_file = max(prediction_files, key=lambda x: os.path.getctime(os.path.join(predictions_dir, x)))
        df = pd.read_csv(os.path.join(predictions_dir, latest_file))
        
        if df.empty:
            st.warning("No predictions data found in the latest file")
            return
        
        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            latest_actual = df['Actual_Price'].iloc[-1]
            st.metric("Latest Price", f"{latest_actual:.2f}")
        
        with col2:
            latest_pred = df['Predicted_Price'].iloc[-1]
            st.metric("Latest Prediction", f"{latest_pred:.2f}")
        
        with col3:
            error_pct = abs(latest_actual - latest_pred) / latest_actual * 100
            st.metric("Current Error %", f"{error_pct:.2f}%")
        
        # Display charts
        st.plotly_chart(create_candlestick_chart(df), use_container_width=True)
        st.plotly_chart(create_metrics_chart(df), use_container_width=True)
        
        # Display recent predictions table
        with st.expander("Recent Predictions", expanded=False):
            st.dataframe(
                df.tail(10).sort_values('Timestamp', ascending=False)
                  .style.format({
                      'Actual_Price': '{:.2f}',
                      'Predicted_Price': '{:.2f}'
                  })
            )
        
    except Exception as e:
        st.error(f"Error loading predictions: {str(e)}")
        st.exception(e)  # This will show the full traceback in development

if __name__ == "__main__":
    realtime_monitoring()