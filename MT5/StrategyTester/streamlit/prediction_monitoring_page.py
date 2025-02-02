import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import time
import os
import logging
from typing import List, Dict, Optional, Tuple

class PredictionMonitoringPage:
    """Class for handling prediction monitoring and visualization in Streamlit."""
    
    def __init__(self, db_path: str):
        """
        Initialize the monitoring page.
        
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
        """
        Check if database and required tables exist.
        
        Returns:
            bool: True if database is properly initialized
        """
        if not os.path.exists(self.db_path):
            return False
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' 
                AND name IN ('prediction_history', 'prediction_metrics')
            """)
            
            existing_tables = {row[0] for row in cursor.fetchall()}
            required_tables = {'prediction_history', 'prediction_metrics'}
            
            return required_tables.issubset(existing_tables)
            
        except sqlite3.Error as e:
            logging.error(f"Database check error: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def get_available_runs(self) -> List[str]:
        """
        Get list of available run IDs from the database.
        
        Returns:
            List of run IDs
        """
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
            return runs
            
        except sqlite3.Error as e:
            logging.error(f"Error getting available runs: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def load_recent_predictions(self, run_id: str, limit: int = 1000) -> pd.DataFrame:
        """
        Load recent predictions from database.
        
        Args:
            run_id: Strategy run identifier
            limit: Maximum number of records to retrieve
            
        Returns:
            DataFrame containing recent predictions
        """
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
        
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn, params=(run_id, limit))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.sort_values('timestamp')
            
        except Exception as e:
            logging.error(f"Error loading predictions: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()

    def load_metrics_history(self, run_id: str) -> pd.DataFrame:
        """
        Load metrics history from database with no caching.
        
        Args:
            run_id: Strategy run identifier
            
        Returns:
            DataFrame containing metrics history
        """
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
        
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn, params=(run_id,), parse_dates=['timestamp'])
            return df.sort_values('timestamp', ascending=True)
            
        except Exception as e:
            logging.error(f"Error loading metrics history: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()

    def get_latest_window_metrics(self, run_id: str, window_size: int = 50) -> pd.DataFrame:
        """
        Get the most recent metrics for a specific window size.
        
        Args:
            run_id: Strategy run identifier
            window_size: Size of the metrics window
            
        Returns:
            DataFrame containing latest metrics
        """
        query = """
            SELECT 
                timestamp,
                rmse,
                mae,
                r2,
                accuracy_rate,
                confident_accuracy_rate
            FROM prediction_metrics
            WHERE run_id = ? AND window_size = ?
            ORDER BY timestamp DESC
            LIMIT 10  -- Get last 10 records for trend calculation
        """
        
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn, params=(run_id, window_size))
            return df
            
        except Exception as e:
            logging.error(f"Error getting latest metrics: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()

    def render_summary_metrics(self, run_id: str, window_size: int = 50):
        """
        Render summary metrics with real-time updates.
        
        Args:
            run_id: Strategy run identifier
            window_size: Size of the metrics window
        """
        df = self.get_latest_window_metrics(run_id, window_size)
        
        if df.empty:
            st.warning("No metrics data available")
            return
            
        # Get the most recent metrics
        latest = df.iloc[0]
        
        # Calculate trends (using the last 10 records)
        if len(df) > 1:
            trends = {
                'rmse': latest['rmse'] - df['rmse'].iloc[-1],
                'mae': latest['mae'] - df['mae'].iloc[-1],
                'r2': latest['r2'] - df['r2'].iloc[-1],
                'accuracy': latest['accuracy_rate'] - df['accuracy_rate'].iloc[-1]
            }
        else:
            trends = {'rmse': 0.0, 'mae': 0.0, 'r2': 0.0, 'accuracy': 0.0}

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "RMSE",
                f"{latest['rmse']:.4f}",
                f"{trends['rmse']:.4f}"
            )
            
        with col2:
            st.metric(
                "MAE",
                f"{latest['mae']:.4f}",
                f"{trends['mae']:.4f}"
            )
            
        with col3:
            st.metric(
                "R²",
                f"{latest['r2']:.4f}",
                f"{trends['r2']:.4f}"
            )
            
        with col4:
            st.metric(
                "Accuracy",
                f"{latest['accuracy_rate']*100:.1f}%",
                f"{trends['accuracy']*100:.1f}%"
            )

    def plot_price_comparison(self, df: pd.DataFrame) -> go.Figure:
        """
        Create price comparison plot.
        
        Args:
            df: DataFrame containing prediction data
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Price comparison plot
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['actual_price'],
                name='Actual Price', 
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['predicted_price'],
                name='Predicted Price', 
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        # Prediction error plot
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['prediction_error'],
                name='Prediction Error', 
                line=dict(color='green')
            ),
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
        """
        Create metrics history plot.
        
        Args:
            df: DataFrame containing metrics history
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RMSE', 'MAE', 'R²', 'Accuracy Rates')
        )
        
        colors = {'10': 'blue', '50': 'red', '100': 'green'}
        
        for window_size in df['window_size'].unique():
            window_df = df[df['window_size'] == window_size]
            name = f'{window_size}-point window'
            
            # RMSE
            fig.add_trace(
                go.Scatter(
                    x=window_df['timestamp'], 
                    y=window_df['rmse'],
                    name=f'RMSE ({name})',
                    line=dict(color=colors[str(window_size)])
                ),
                row=1, col=1
            )
            
            # MAE
            fig.add_trace(
                go.Scatter(
                    x=window_df['timestamp'], 
                    y=window_df['mae'],
                    name=f'MAE ({name})',
                    line=dict(color=colors[str(window_size)])
                ),
                row=1, col=2
            )
            
            # R²
            fig.add_trace(
                go.Scatter(
                    x=window_df['timestamp'], 
                    y=window_df['r2'],
                    name=f'R² ({name})',
                    line=dict(color=colors[str(window_size)])
                ),
                row=2, col=1
            )
            
            # Accuracy Rates
            fig.add_trace(
                go.Scatter(
                    x=window_df['timestamp'], 
                    y=window_df['accuracy_rate'],
                    name=f'Accuracy ({name})',
                    line=dict(color=colors[str(window_size)], dash='solid')
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=window_df['timestamp'], 
                    y=window_df['confident_accuracy_rate'],
                    name=f'Confident Accuracy ({name})',
                    line=dict(color=colors[str(window_size)], dash='dot')
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title='Prediction Metrics History',
            showlegend=True
        )
        
        return fig

    def get_aggregate_statistics(self, run_id: str) -> pd.DataFrame:
        """
        Calculate enhanced aggregate statistics including multiple accuracy metrics.
        """
        query = """
            WITH price_stats AS (
                SELECT 
                    model_name,
                    COUNT(*) as total_predictions,
                    AVG(ABS(prediction_error)) as mean_absolute_error,
                    AVG(prediction_error * prediction_error) as mean_squared_error,
                    AVG(CASE WHEN is_confident = 1 THEN 1 ELSE 0 END) as confident_prediction_rate,
                    MIN(actual_price) as min_actual,
                    MAX(actual_price) as max_actual,
                    MIN(predicted_price) as min_predicted,
                    MAX(predicted_price) as max_predicted,
                    AVG(confidence) as avg_confidence,
                    AVG((predicted_price - actual_price) * (predicted_price - actual_price)) as mse,
                    AVG(actual_price * actual_price) - (AVG(actual_price) * AVG(actual_price)) as var_actual,
                    -- Add prediction accuracy at different thresholds
                    AVG(CASE 
                        WHEN ABS(prediction_error) <= 0.01 * actual_price THEN 1.0 
                        ELSE 0.0 
                    END) as accuracy_1pct,
                    AVG(CASE 
                        WHEN ABS(prediction_error) <= 0.02 * actual_price THEN 1.0 
                        ELSE 0.0 
                    END) as accuracy_2pct,
                    AVG(CASE 
                        WHEN ABS(prediction_error) <= 0.05 * actual_price THEN 1.0 
                        ELSE 0.0 
                    END) as accuracy_5pct
                FROM prediction_history
                WHERE run_id = ?
                GROUP BY model_name
            ),
            confident_metrics AS (
                SELECT 
                    model_name,
                    AVG(CASE 
                        WHEN ABS(prediction_error) <= 0.05 * actual_price THEN 1.0 
                        ELSE 0.0 
                    END) as confident_success_rate
                FROM prediction_history
                WHERE run_id = ? AND is_confident = 1
                GROUP BY model_name
            )
            SELECT 
                p.*,
                SQRT(p.mean_squared_error) as rmse,
                1 - (p.mse / NULLIF(p.var_actual, 0)) as r_squared,
                COALESCE(c.confident_success_rate, 0) as confident_success_rate
            FROM price_stats p
            LEFT JOIN confident_metrics c ON p.model_name = c.model_name
        """
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # First, get the main statistics
            df = pd.read_sql_query(query, conn, params=(run_id, run_id))
            
            if not df.empty:
                # Calculate directional accuracy separately for each model
                for index, row in df.iterrows():
                    direction_query = """
                        WITH ordered_prices AS (
                            SELECT 
                                actual_price,
                                predicted_price,
                                actual_price - LAG(actual_price) OVER (ORDER BY timestamp) as price_change
                            FROM prediction_history
                            WHERE run_id = ? AND model_name = ?
                            ORDER BY timestamp
                        )
                        SELECT 
                            AVG(CASE 
                                WHEN (price_change > 0 AND predicted_price > actual_price) OR
                                     (price_change < 0 AND predicted_price < actual_price)
                                THEN 1.0 
                                ELSE 0.0 
                            END) as direction_accuracy
                        FROM ordered_prices
                        WHERE price_change IS NOT NULL
                    """
                    cursor.execute(direction_query, (run_id, row['model_name']))
                    direction_accuracy = cursor.fetchone()[0] or 0
                    df.at[index, 'directional_accuracy'] = direction_accuracy
                
                # Calculate Information Ratio
                df['information_ratio'] = (df['confident_success_rate'] - 0.5) / \
                                        df['rmse'].where(df['rmse'] != 0, np.inf)
                
                # Format percentages
                percentage_cols = [
                    'confident_prediction_rate', 'avg_confidence', 
                    'directional_accuracy', 'confident_success_rate',
                    'accuracy_1pct', 'accuracy_2pct', 'accuracy_5pct'
                ]
                for col in percentage_cols:
                    df[col] = df[col] * 100
                
                # Reorder columns
                columns = [
                    'model_name',
                    'total_predictions',
                    'rmse',
                    'mean_absolute_error',
                    'r_squared',
                    'accuracy_1pct',
                    'accuracy_2pct',
                    'accuracy_5pct',
                    'directional_accuracy',
                    'confident_success_rate',
                    'confident_prediction_rate',
                    'avg_confidence',
                    'information_ratio',
                    'min_actual',
                    'max_actual',
                    'min_predicted',
                    'max_predicted'
                ]
                df = df[columns]
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating aggregate statistics: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def display_aggregate_statistics(self, run_id: str):
        """Display enhanced aggregate statistics in a formatted table."""
        df = self.get_aggregate_statistics(run_id)
        
        if df.empty:
            st.warning("No aggregate statistics available")
            return
            
        # Format the DataFrame for display
        display_df = df.copy()
        display_df = display_df.round(4)
        
        # Rename columns for better display
        display_df.columns = [
            'Model Name',
            'Total Predictions',
            'RMSE',
            'MAE',
            'R²',
            'Accuracy (1%)',
            'Accuracy (2%)',
            'Accuracy (5%)',
            'Directional Accuracy (%)',
            'Confident Success Rate (%)',
            'Confident Pred. Rate (%)',
            'Avg. Confidence (%)',
            'Information Ratio',
            'Min Actual',
            'Max Actual',
            'Min Predicted',
            'Max Predicted'
        ]
        
        st.subheader("Aggregate Model Performance")
        st.dataframe(
            display_df.style.format({
                'RMSE': '{:.4f}',
                'MAE': '{:.4f}',
                'R²': '{:.4f}',
                'Accuracy (1%)': '{:.2f}%',
                'Accuracy (2%)': '{:.2f}%',
                'Accuracy (5%)': '{:.2f}%',
                'Directional Accuracy (%)': '{:.2f}%',
                'Confident Success Rate (%)': '{:.2f}%',
                'Confident Pred. Rate (%)': '{:.2f}%',
                'Avg. Confidence (%)': '{:.2f}%',
                'Information Ratio': '{:.4f}',
                'Min Actual': '{:.4f}',
                'Max Actual': '{:.4f}',
                'Min Predicted': '{:.4f}',
                'Max Predicted': '{:.4f}'
            }),
            use_container_width=True
        )
        
        # Add metric explanations
        with st.expander("Metric Explanations"):
            st.markdown("""
            ### Metric Definitions
            - **RMSE**: Root Mean Square Error - measures prediction accuracy
            - **MAE**: Mean Absolute Error - average absolute difference between predictions and actual values
            - **R²**: Coefficient of determination (0-1) - how well the model explains price variations
            - **Accuracy**: Percentage of predictions within X% of actual price (1%, 2%, 5% thresholds)
            - **Directional Accuracy**: Percentage of correct predictions in terms of price movement direction
            - **Confident Success Rate**: Success rate when the model is confident (within 5% margin)
            - **Information Ratio**: Risk-adjusted performance metric (higher is better)
            
            ### Interpretation Guide
            - **RMSE & MAE**: Lower values indicate better accuracy
            - **R²**: Closer to 1 is better (above 0.7 is typically good)
            - **Accuracy**: Higher percentages indicate better prediction accuracy
            - **Directional Accuracy**: Above 50% indicates better than random
            - **Information Ratio**: Higher values indicate better risk-adjusted performance
            - **Confident Pred. Rate**: Higher values show more decisive model behavior
            """)

def prediction_monitoring_page():
    # Add CSS for tooltip
    st.markdown("""
        <style>
        .header-container {
            display: flex;
            align-items: center;
            gap: 10px;
            position: relative;
        }
        .info-icon {
            color: #00ADB5;
            font-size: 1.2rem;
            cursor: help;
            text-decoration: none;
            position: relative;
            display: inline-block;
        }
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 500px;
            background-color: #252830;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 15px;
            position: absolute;
            z-index: 9999;
            top: -10px;
            left: 30px;
            opacity: 0;
            transition: opacity 0.3s;
            border: 1px solid #333;
            font-size: 0.9rem;
            line-height: 1.6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header with info icon
    st.markdown("""
        <div class='header-container'>
            <h2 style='color: #00ADB5; padding: 1rem 0; margin: 0;'>
                Real-Time Prediction Monitoring
            </h2>
            <div class='tooltip'>
                <span class='info-icon'>ℹ️</span>
                <div class='tooltiptext'>
                    We are using the prediction_history table for real time prediction
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Initialize monitoring
    db_path = "logs/trading_data.db"
    monitor = PredictionMonitoringPage(db_path)
    
    # Check if database is initialized
    if not monitor.check_database_exists():
        st.warning("""
        Prediction monitoring database not initialized. 
        Please start the ZMQ server and make some predictions first.
        """)
        return
    
    # Get available runs
    runs = monitor.get_available_runs()
    if not runs:
        st.info("No prediction data available yet.")
        return
    
    # Sidebar controls
    st.sidebar.header("Controls")
    selected_run = st.sidebar.selectbox("Select Run ID", runs)
    window_size = st.sidebar.selectbox(
        "Metrics Window Size",
        [10, 50, 100],
        index=1  # Default to 50
    )
    
    update_interval = st.sidebar.number_input(
        "Update Interval (seconds)",
        min_value=1,
        max_value=60,
        value=5
    )
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    
    try:
        # Display aggregate statistics first
        monitor.display_aggregate_statistics(selected_run)
        
        # Display summary metrics for selected window size
        st.subheader(f"Summary Metrics (Window Size: {window_size})")
        monitor.render_summary_metrics(selected_run, window_size)
        
        # Load data for charts
        predictions_df = monitor.load_recent_predictions(selected_run)
        
        if not predictions_df.empty:
            # Display plots
            st.subheader("Price Comparison")
            price_fig = monitor.plot_price_comparison(predictions_df)
            st.plotly_chart(price_fig, use_container_width=True)
            
            st.subheader("Metrics History")
            metrics_df = monitor.load_metrics_history(selected_run)
            if not metrics_df.empty:
                metrics_fig = monitor.plot_metrics_history(metrics_df)
                st.plotly_chart(metrics_fig, use_container_width=True)
        else:
            st.warning("No predictions found for selected run.")
        
        # Auto-refresh the page
        if auto_refresh:
            time.sleep(update_interval)
            st.rerun()
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logging.error(f"Error in prediction monitoring page: {str(e)}")
