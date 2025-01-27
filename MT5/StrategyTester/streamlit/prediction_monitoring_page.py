import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import time
import os

class PredictionMonitoringPage:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def check_database_exists(self) -> bool:
        """Check if database and required tables exist"""
        if not os.path.exists(self.db_path):
            return False
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if required tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' 
                AND name IN ('prediction_history', 'prediction_metrics')
            """)
            
            existing_tables = {row[0] for row in cursor.fetchall()}
            required_tables = {'prediction_history', 'prediction_metrics'}
            
            return required_tables.issubset(existing_tables)
            
        except sqlite3.Error:
            return False
        finally:
            if conn:
                conn.close()
        
    def load_recent_predictions(self, run_id: str, limit: int = 1000) -> pd.DataFrame:
        """Load recent predictions from database"""
        query = """
            SELECT 
                timestamp,
                actual_price,
                predicted_price,
                confidence,
                model_name,
                prediction_error,
                is_confident
            FROM prediction_history
            WHERE run_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(query, conn, params=(run_id, limit))
        conn.close()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
        
    def load_metrics_history(self, run_id: str) -> pd.DataFrame:
        """Load metrics history from database"""
        query = """
            SELECT 
                timestamp,
                window_size,
                rmse,
                mae,
                r2,
                accuracy_rate,
                confident_accuracy_rate
            FROM prediction_metrics
            WHERE run_id = ?
            ORDER BY timestamp DESC
        """
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(query, conn, params=(run_id,))
        conn.close()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
        
    # def get_available_runs(self) -> list:
    #     """Get list of available run_ids"""
    #     query = """
    #         SELECT DISTINCT run_id 
    #         FROM prediction_history 
    #         ORDER BY run_id
    #     """
        
    #     conn = sqlite3.connect(self.db_path)
    #     cursor = conn.cursor()
    #     cursor.execute(query)
    #     runs = [row[0] for row in cursor.fetchall()]
    #     conn.close()
        
    #     return runs


    def get_available_runs(self) -> list:
        """Get list of available run_ids"""
        if not self.check_database_exists():
            return []
            
        try:
            query = """
                SELECT DISTINCT run_id 
                FROM prediction_history 
                ORDER BY run_id
            """
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            runs = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return runs
            
        except sqlite3.Error:
            return []
        

    def plot_price_comparison(self, df: pd.DataFrame) -> go.Figure:
        """Create price comparison plot"""
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.1,
                           row_heights=[0.7, 0.3])
        
        # Price comparison plot
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['actual_price'],
                      name='Actual Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['predicted_price'],
                      name='Predicted Price', line=dict(color='red')),
            row=1, col=1
        )
        
        # Prediction error plot
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['prediction_error'],
                      name='Prediction Error', line=dict(color='green')),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            title='Price Prediction Comparison',
            xaxis2_title='Time',
            yaxis_title='Price',
            yaxis2_title='Error'
        )
        
        return fig
        
    def plot_metrics_history(self, df: pd.DataFrame) -> go.Figure:
        """Create metrics history plot"""
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('RMSE', 'MAE', 'R²', 'Accuracy Rates'))
        
        colors = {'10': 'blue', '50': 'red', '100': 'green'}
        
        for window_size in df['window_size'].unique():
            window_df = df[df['window_size'] == window_size]
            name = f'{window_size}-point window'
            
            # RMSE
            fig.add_trace(
                go.Scatter(x=window_df['timestamp'], y=window_df['rmse'],
                          name=f'RMSE ({name})',
                          line=dict(color=colors[str(window_size)])),
                row=1, col=1
            )
            
            # MAE
            fig.add_trace(
                go.Scatter(x=window_df['timestamp'], y=window_df['mae'],
                          name=f'MAE ({name})',
                          line=dict(color=colors[str(window_size)])),
                row=1, col=2
            )
            
            # R²
            fig.add_trace(
                go.Scatter(x=window_df['timestamp'], y=window_df['r2'],
                          name=f'R² ({name})',
                          line=dict(color=colors[str(window_size)])),
                row=2, col=1
            )
            
            # Accuracy Rates
            fig.add_trace(
                go.Scatter(x=window_df['timestamp'], y=window_df['accuracy_rate'],
                          name=f'Accuracy ({name})',
                          line=dict(color=colors[str(window_size)], dash='solid')),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=window_df['timestamp'], y=window_df['confident_accuracy_rate'],
                          name=f'Confident Accuracy ({name})',
                          line=dict(color=colors[str(window_size)], dash='dot')),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title='Prediction Metrics History',
            showlegend=True
        )
        
        return fig

    def render_summary_metrics(self, df: pd.DataFrame):
        """Render summary metrics in Streamlit"""
        latest_metrics = df.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Latest RMSE",
                f"{latest_metrics['rmse']:.4f}",
                delta=f"{latest_metrics['rmse'] - df['rmse'].mean():.4f}"
            )
            
        with col2:
            st.metric(
                "Latest MAE",
                f"{latest_metrics['mae']:.4f}",
                delta=f"{latest_metrics['mae'] - df['mae'].mean():.4f}"
            )
            
        with col3:
            st.metric(
                "Latest R²",
                f"{latest_metrics['r2']:.4f}",
                delta=f"{latest_metrics['r2'] - df['r2'].mean():.4f}"
            )
            
        with col4:
            st.metric(
                "Latest Accuracy",
                f"{latest_metrics['accuracy_rate']*100:.1f}%",
                delta=f"{(latest_metrics['accuracy_rate'] - df['accuracy_rate'].mean())*100:.1f}%"
            )

def prediction_monitoring_page():
    st.title("Real-Time Prediction Monitoring")

    
    
    # Initialize monitoring
    db_path = "logs/trading_data.db"  # Adjust path as needed

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    
    monitor = PredictionMonitoringPage(db_path)

    # Check if database is initialized
    if not monitor.check_database_exists():
        st.warning("""
        Prediction monitoring database not initialized. 
        Please start the ZMQ server and make some predictions first.
        
        Steps to get started:
        1. Go to 'ZMQ Server Control' page
        2. Start the ZMQ server
        3. Run your MT5 strategy to generate predictions
        4. Return to this page to see the monitoring
        """)
        return
    
    # Get available runs
    runs = monitor.get_available_runs()
    if not runs:
        # st.error("No prediction data available. Please run some predictions first.")
        st.info("""
        No prediction data available yet. 
        
        This could mean:
        - The ZMQ server hasn't received any prediction requests
        - Your MT5 strategy hasn't started sending data
        - The predictions haven't been recorded yet
        
        Please ensure:
        1. ZMQ server is running
        2. Your MT5 strategy is active
        3. Data is being sent to the server
        """)
        return
        
    # Sidebar controls
    st.sidebar.header("Controls")
    selected_run = st.sidebar.selectbox("Select Run ID", runs)
    
    update_interval = st.sidebar.number_input(
        "Update Interval (seconds)",
        min_value=1,
        max_value=60,
        value=5
    )
    
    # Auto-refresh checkbox
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    
    # Main content
    try:
        # Load data
        predictions_df = monitor.load_recent_predictions(selected_run)
        metrics_df = monitor.load_metrics_history(selected_run)
        
        if predictions_df.empty:
            st.warning("No predictions found for selected run.")
            return
            
        # Display summary metrics
        st.subheader("Summary Metrics")
        latest_window = metrics_df[metrics_df['window_size'] == 50].copy()
        if not latest_window.empty:
            monitor.render_summary_metrics(latest_window)
            
        # Display plots
        st.subheader("Price Comparison")
        price_fig = monitor.plot_price_comparison(predictions_df)
        st.plotly_chart(price_fig, use_container_width=True)
        
        st.subheader("Metrics History")
        metrics_fig = monitor.plot_metrics_history(metrics_df)
        st.plotly_chart(metrics_fig, use_container_width=True)
        
        # Add model performance analysis
        st.subheader("Model Performance Analysis")
        model_metrics = predictions_df.groupby('model_name').agg({
            'prediction_error': ['mean', 'std'],
            'confidence': 'mean',
            'is_confident': 'mean'
        }).round(4)
        
        st.dataframe(model_metrics)
        
        # # Auto-refresh logic
        # if auto_refresh:
        #     st.empty()
        #     st.experimental_rerun()
        # Auto-refresh logic
        if auto_refresh:
            time.sleep(update_interval)
            st.rerun()
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")