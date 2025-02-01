import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

class HistoricalPredictionsPage:
    """Class for analyzing historical predictions in Streamlit."""
    
    def __init__(self, db_path: str):
        """
        Initialize the analysis page.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def check_database_exists(self) -> bool:
        """Check if database and required tables exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' 
                AND name IN ('historical_predictions', 'historical_prediction_metrics')
            """)
            
            existing_tables = {row[0] for row in cursor.fetchall()}
            required_tables = {'historical_predictions', 'historical_prediction_metrics'}
            
            return required_tables.issubset(existing_tables)
            
        except sqlite3.Error as e:
            logging.error(f"Database check error: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def get_available_runs(self) -> List[Dict[str, str]]:
        """Get list of available runs with metadata."""
        try:
            query = """
                SELECT DISTINCT 
                    run_id,
                    source_table,
                    model_name,
                    MIN(datetime) as start_date,
                    MAX(datetime) as end_date,
                    COUNT(*) as prediction_count
                FROM historical_predictions
                GROUP BY run_id, source_table, model_name
                ORDER BY datetime DESC
            """
            
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn)
            return df.to_dict('records')
            
        except sqlite3.Error as e:
            logging.error(f"Error getting available runs: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def load_predictions(self, run_id: str) -> pd.DataFrame:
        """Load predictions for a specific run."""
        query = """
            SELECT 
                datetime,
                actual_price,
                predicted_price,
                error,
                price_change,
                predicted_change,
                price_volatility
            FROM historical_predictions
            WHERE run_id = ?
            ORDER BY datetime
        """
        
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn, params=(run_id,))
            df['datetime'] = pd.to_datetime(df['datetime'])
            return df
            
        except Exception as e:
            logging.error(f"Error loading predictions: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()

    def load_metrics(self, run_id: str) -> pd.DataFrame:
        """Load detailed metrics for a specific run."""
        query = """
            SELECT *
            FROM historical_prediction_metrics
            WHERE run_id = ?
            ORDER BY timestamp
        """
        
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn, params=(run_id,))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
            
        except Exception as e:
            logging.error(f"Error loading metrics: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()

    def plot_predictions(self, df: pd.DataFrame) -> List[go.Figure]:
        """Create comprehensive prediction analysis plots."""
        # Convert datetime index to pandas datetime if not already
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Sort by datetime to ensure proper line connection
        df = df.sort_values('datetime')
        
        # Get unique dates for vertical lines
        unique_dates = df['datetime'].dt.date.unique()
        
        # Create figure
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                'Price Comparison',
                'Prediction Error',
                'Price Volatility'
            ),
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Price comparison
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df['actual_price'],
                name='Actual Price',
                line=dict(color='blue'),
                mode='lines',
                connectgaps=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df['predicted_price'],
                name='Predicted Price',
                line=dict(color='red'),
                mode='lines',
                connectgaps=False
            ),
            row=1, col=1
        )
        
        # Prediction error
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df['error'],
                name='Prediction Error',
                line=dict(color='orange'),
                mode='lines',
                connectgaps=False
            ),
            row=2, col=1
        )
        
        # Price volatility
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df['price_volatility'],
                name='Price Volatility',
                line=dict(color='purple'),
                mode='lines',
                connectgaps=False
            ),
            row=3, col=1
        )
        
        # Add vertical lines for each day
        for date in unique_dates:
            date_str = date.strftime('%Y-%m-%d')
            # Add vertical line to each subplot
            for row in range(1, 4):
                fig.add_vline(
                    x=date_str + " 16:30",  # Start of trading day
                    line_width=1,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5,
                    row=row,
                    col=1
                )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title='Historical Prediction Analysis',
        )
        
        # Update x-axes to compress the view and show dates consistently
        for i in range(1, 4):
            fig.update_xaxes(
                row=i,
                col=1,
                rangebreaks=[
                    dict(bounds=[23, 16.5], pattern="hour"),  # Hide hours outside trading time
                    dict(bounds=["sat", "mon"])  # Hide weekends
                ],
                tickformat="%Y-%m-%d %H:%M",
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                dtick="D1",  # Show tick for each day
                tickmode="auto",
                tickangle=45,  # Angle the date labels for better readability
            )
        
        # Update y-axes grid and zeroline
        fig.update_yaxes(
            gridcolor='rgba(128, 128, 128, 0.2)',
            zerolinecolor='rgba(128, 128, 128, 0.2)'
        )
        
        return [fig]

    def plot_metrics_history(self, df: pd.DataFrame) -> List[go.Figure]:
        """Create detailed metrics history visualization."""
        # Convert timestamp to datetime if not already
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Error Metrics',
                'Directional Accuracy',
                'Movement Analysis',
                'Streak Analysis'
            )
        )
        
        # Error metrics
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['mean_absolute_error'],
                name='MAE', 
                line=dict(color='blue'),
                mode='lines',
                connectgaps=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['root_mean_squared_error'],
                name='RMSE', 
                line=dict(color='red'),
                mode='lines',
                connectgaps=False
            ),
            row=1, col=1
        )
        
        # Directional accuracy
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['direction_accuracy'],
                name='Direction Accuracy', 
                line=dict(color='green'),
                mode='lines',
                connectgaps=False
            ),
            row=1, col=2
        )
        
        # Movement analysis
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['up_prediction_accuracy'],
                name='Up Accuracy', 
                line=dict(color='purple'),
                mode='lines',
                connectgaps=False
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['down_prediction_accuracy'],
                name='Down Accuracy', 
                line=dict(color='orange'),
                mode='lines',
                connectgaps=False
            ),
            row=2, col=1
        )
        
        # Streak analysis
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['max_correct_streak'],
                name='Max Streak', 
                line=dict(color='brown'),
                mode='lines',
                connectgaps=False
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['avg_correct_streak'],
                name='Avg Streak', 
                line=dict(color='pink'),
                mode='lines',
                connectgaps=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title='Metrics History Analysis'
        )
        
        # Update x-axes to compress the view
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=[23, 16.5], pattern="hour"),  # Hide hours outside trading time
                dict(bounds=["sat", "mon"])  # Hide weekends
            ],
            tickformat="%Y-%m-%d %H:%M"
        )
        
        return [fig]

def historical_predictions_page():
    """Main function for the historical predictions analysis page."""
    st.title("Historical Predictions Analysis")
    
    # Initialize analysis
    db_path = "logs/trading_data.db"
    analyzer = HistoricalPredictionsPage(db_path)
    
    # Check if database is initialized
    if not analyzer.check_database_exists():
        st.warning("""
        Historical predictions database not found. 
        Please run predictions using run_predictions.py first.
        """)
        return
    
    # Get available runs
    runs = analyzer.get_available_runs()
    if not runs:
        st.info("No historical prediction data available.")
        return
    
    # Run selection with metadata
    st.sidebar.header("Analysis Controls")
    
    # Create a formatted selection list
    run_options = [
        f"{run['run_id']} - {run['model_name']} ({run['source_table']})" 
        for run in runs
    ]
    
    selected_option = st.sidebar.selectbox(
        "Select Prediction Run",
        options=run_options
    )
    
    # Extract run_id from selection
    selected_run_id = selected_option.split(' - ')[0]
    
    # Get selected run metadata
    selected_run = next(run for run in runs if run['run_id'] == selected_run_id)
    
    # Display run information
    with st.expander("Run Information", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", selected_run['model_name'])
        with col2:
            st.metric("Data Source", selected_run['source_table'])
        with col3:
            st.metric("Predictions", selected_run['prediction_count'])
        
        st.write(f"Period: {selected_run['start_date']} to {selected_run['end_date']}")
    
    try:
        # Load data
        predictions_df = analyzer.load_predictions(selected_run_id)
        metrics_df = analyzer.load_metrics(selected_run_id)
        
        if not predictions_df.empty:
            # Display predictions plots
            st.subheader("Prediction Analysis")
            prediction_figs = analyzer.plot_predictions(predictions_df)
            for fig in prediction_figs:
                st.plotly_chart(fig, use_container_width=True)
            
            if not metrics_df.empty:
                # Display metrics plots
                st.subheader("Metrics Analysis")
                metrics_figs = analyzer.plot_metrics_history(metrics_df)
                for fig in metrics_figs:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display latest metrics
                latest_metrics = metrics_df.iloc[-1]
                
                st.subheader("Final Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("RMSE", f"{latest_metrics['root_mean_squared_error']:.4f}")
                with col2:
                    st.metric("MAE", f"{latest_metrics['mean_absolute_error']:.4f}")
                with col3:
                    st.metric("Direction Accuracy", f"{latest_metrics['direction_accuracy']*100:.1f}%")
                with col4:
                    st.metric("Max Correct Streak", int(latest_metrics['max_correct_streak']))
            
            # Add download options
            st.sidebar.subheader("Download Data")
            
            csv_predictions = predictions_df.to_csv(index=False)
            st.sidebar.download_button(
                label="Download Predictions",
                data=csv_predictions,
                file_name=f"predictions_{selected_run_id}.csv",
                mime="text/csv"
            )
            
            if not metrics_df.empty:
                csv_metrics = metrics_df.to_csv(index=False)
                st.sidebar.download_button(
                    label="Download Metrics",
                    data=csv_metrics,
                    file_name=f"metrics_{selected_run_id}.csv",
                    mime="text/csv"
                )
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logging.error(f"Error in historical predictions page: {str(e)}")

