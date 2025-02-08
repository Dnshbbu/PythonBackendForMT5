import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
# Import the required functions from model_comparison_page
from model_comparison_page import display_model_details_section, ModelComparison

# Constants for optimization
ROWS_PER_PAGE = 1000
CACHE_TTL = 3600  # Cache time to live in seconds

@st.cache_data(ttl=CACHE_TTL)
def get_available_runs(db_path: str) -> List[Dict[str, str]]:
    """Get list of available runs with metadata."""
    try:
        query = """
            SELECT DISTINCT 
                p.run_id,
                p.source_table,
                p.model_name,
                MIN(p.datetime) as start_date,
                MAX(p.datetime) as end_date,
                COUNT(*) as prediction_count,
                MAX(m.timestamp) as run_timestamp
            FROM historical_predictions p
            LEFT JOIN historical_prediction_metrics m 
            ON p.run_id = m.run_id
            GROUP BY p.run_id, p.source_table, p.model_name
            ORDER BY p.datetime DESC
        """
        
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        return df.to_dict('records')
        
    except sqlite3.Error as e:
        logging.error(f"Error getting available runs: {e}")
        return []
    finally:
        if conn:
            conn.close()

@st.cache_data(ttl=CACHE_TTL)
def load_predictions_page(db_path: str, run_id: str, page: int) -> pd.DataFrame:
    """Load predictions for a specific run with pagination."""
    offset = page * ROWS_PER_PAGE
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
        LIMIT ? OFFSET ?
    """
    
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn, params=(run_id, ROWS_PER_PAGE, offset))
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
        
    except Exception as e:
        logging.error(f"Error loading predictions: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

@st.cache_data(ttl=CACHE_TTL)
def get_total_rows(db_path: str, run_id: str) -> int:
    """Get total number of rows for a run."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM historical_predictions WHERE run_id = ?", 
            (run_id,)
        )
        return cursor.fetchone()[0]
    except sqlite3.Error as e:
        logging.error(f"Error getting total rows: {e}")
        return 0
    finally:
        if conn:
            conn.close()

@st.cache_data(ttl=CACHE_TTL)
def load_metrics(db_path: str, run_id: str) -> pd.DataFrame:
    """Load detailed metrics for a specific run."""
    query = """
        SELECT *
        FROM historical_prediction_metrics
        WHERE run_id = ?
        ORDER BY timestamp
    """
    
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn, params=(run_id,))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
        
    except Exception as e:
        logging.error(f"Error loading metrics: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

@st.cache_data(ttl=CACHE_TTL)
def load_all_predictions(db_path: str, run_id: str) -> pd.DataFrame:
    """Load all predictions for a specific run."""
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
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn, params=(run_id,))
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
        
    except Exception as e:
        logging.error(f"Error loading all predictions: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

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
                    p.run_id,
                    p.source_table,
                    p.model_name,
                    MIN(p.datetime) as start_date,
                    MAX(p.datetime) as end_date,
                    COUNT(*) as prediction_count,
                    MAX(m.timestamp) as run_timestamp
                FROM historical_predictions p
                LEFT JOIN historical_prediction_metrics m 
                ON p.run_id = m.run_id
                GROUP BY p.run_id, p.source_table, p.model_name
                ORDER BY p.datetime DESC
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

    def plot_predictions(self, df: pd.DataFrame, max_points: int = None) -> List[go.Figure]:
        """Create comprehensive prediction analysis plots."""
        # Downsample data for plotting if max_points is specified
        if max_points and len(df) > max_points:
            df = df.iloc[::len(df)//max_points]  # Take max_points evenly spaced samples
        
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

def format_metric_value(name: str, value: float) -> str:
    """Format metric values based on their type/name."""
    if 'accuracy' in name.lower() or 'ratio' in name.lower() or 'rate' in name.lower():
        return f"{value*100:.1f}%"
    elif 'streak' in name.lower() and 'max' in name.lower():
        return f"{int(value)}"
    elif isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)

def format_column_name(col_name: str) -> str:
    """Convert snake_case column names to Title Case for display."""
    return ' '.join(word.capitalize() for word in col_name.split('_'))

def historical_predictions_page():
    """Main function for the historical predictions analysis page."""
    # Add CSS for tooltip and styling
    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
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
                Historical Predictions Analysis
            </h2>
            <div class='tooltip'>
                <span class='info-icon'>ℹ️</span>
                <div class='tooltiptext'>
                    <div style='margin-bottom: 10px;'>
                        We are comparing the predictions run on historical data stored in the tables.
                    </div>
                    <div>
                        Other tables that are used - "model_repository" to get model details.
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Initialize session state for sidebar toggle
    if 'show_details_sidebar' not in st.session_state:
        st.session_state.show_details_sidebar = False

    # Initialize analysis
    db_path = "logs/trading_data.db"
    analyzer = HistoricalPredictionsPage(db_path)
    
    # Check if database is initialized
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            AND name IN ('historical_predictions', 'historical_prediction_metrics')
        """)
        existing_tables = {row[0] for row in cursor.fetchall()}
        required_tables = {'historical_predictions', 'historical_prediction_metrics'}
        
        if not required_tables.issubset(existing_tables):
            st.warning("""
            Historical predictions database not found. 
            Please run predictions using run_predictions.py first.
            """)
            return
    except sqlite3.Error as e:
        st.error(f"Database error: {str(e)}")
        return
    finally:
        if conn:
            conn.close()

    # Get available runs
    runs = get_available_runs(db_path)
    if not runs:
        st.info("No historical prediction data available.")
        return

    # Left sidebar toggle button
    with st.sidebar:
        st.header("Analysis Controls")
        st.button(
            "Show Model Details",
            key="toggle_details",
            on_click=lambda: setattr(st.session_state, 'show_details_sidebar', 
                                   not st.session_state.show_details_sidebar)
        )

    # Main content and details sidebar layout
    if st.session_state.show_details_sidebar:
        main_col, details_col = st.columns([3, 1])
    else:
        main_col = st.container()
        details_col = None

    with main_col:
        # Run selection with metadata
        run_options = [
            f"{run['run_id']} - {run['model_name']} ({run['source_table']})" 
            for run in runs
        ]
        
        selected_option = st.selectbox(
            "Select Prediction Run",
            options=run_options
        )
        
        # Extract run_id from selection
        selected_run_id = selected_option.split(' - ')[0]
        
        # Get selected run metadata
        selected_run = next(run for run in runs if run['run_id'] == selected_run_id)

        # Display run information
        with st.expander("Run Information", expanded=False):
            st.markdown("""
                <style>
                    .small-font {
                        font-size: 14px;
                    }
                    .metric-label {
                        font-size: 12px;
                        color: #808495;
                    }
                    .metric-value {
                        font-size: 16px;
                    }
                    .info-separator {
                        margin: 0 15px;
                        color: #808495;
                    }
                    .info-label {
                        color: #808495;
                    }
                    .info-value {
                        color: #FFFFFF;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<p class="metric-label">Model</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{selected_run["model_name"]}</p>', unsafe_allow_html=True)
            with col2:
                st.markdown('<p class="metric-label">Data Source</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{selected_run["source_table"]}</p>', unsafe_allow_html=True)
            with col3:
                st.markdown('<p class="metric-label">Predictions</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{selected_run["prediction_count"]}</p>', unsafe_allow_html=True)
            
            run_time = pd.to_datetime(selected_run["run_timestamp"]).strftime("%Y-%m-%d %H:%M")
            st.markdown(
                f'<p class="small-font"><span class="info-label">Period:</span> <span class="info-value">{selected_run["start_date"]} to {selected_run["end_date"]}</span> <span class="info-separator">|</span> <span class="info-label">Run Time:</span> <span class="info-value">{run_time}</span></p>', 
                unsafe_allow_html=True
            )
        
        try:
            # Get total rows and calculate total pages
            total_rows = get_total_rows(db_path, selected_run_id)
            total_pages = max(1, (total_rows + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)
            
            # Initialize page from session state
            if 'current_page' not in st.session_state:
                st.session_state['current_page'] = 0
            page = st.session_state['current_page']
            
            # Add view options
            view_option = st.radio(
                "Select View Mode",
                ["Paginated View", "Full Data View"],
                help="Paginated View loads data in chunks. Full Data View loads all data at once (may be slower for large datasets)."
            )
            
            if view_option == "Paginated View":
                # Ensure we have valid page numbers
                max_page = max(0, total_pages - 1)  # Ensure at least 0
                
                # Handle case where there's only one page
                if max_page == 0:
                    page = 0
                    st.info("Only one page of data available")
                else:
                    # Show navigation buttons for pagination
                    st.write("### Page Navigation")
                    col1, col2, col3 = st.columns([1, 3, 1])
                    
                    with col1:
                        if st.button("← Previous", disabled=(page <= 0)):
                            page = max(0, page - 1)
                            st.session_state['current_page'] = page
                            st.rerun()
                    
                    with col2:
                        st.write(f"Page {page + 1} of {total_pages}")
                    
                    with col3:
                        if st.button("Next →", disabled=(page >= max_page)):
                            page = min(max_page, page + 1)
                            st.session_state['current_page'] = page
                            st.rerun()
                
                # Ensure page is within valid range
                page = min(max(0, page), max_page)
                st.session_state['current_page'] = page
                
                # Load data for current page
                predictions_df = load_predictions_page(db_path, selected_run_id, page)
                
                if predictions_df.empty:
                    st.warning("No predictions found for the selected page.")
                    return
                
                page_info = f"(Page {page + 1} of {total_pages})"
            
            else:  # Full Data View
                predictions_df = load_all_predictions(db_path, selected_run_id)
                
                if predictions_df.empty:
                    st.warning("No predictions found.")
                    return
                
                page_info = f"(Showing all {len(predictions_df):,} records)"
            
            # Load metrics (same for both views)
            metrics_df = load_metrics(db_path, selected_run_id)
            
            # Display metrics first if available
            if not metrics_df.empty:
                with st.expander("Performance Metrics", expanded=False):
                    st.subheader("Performance Metrics")
                    
                    # Get the latest metrics
                    display_metrics = metrics_df.iloc[-1].copy()
                    
                    # Exclude non-metric columns if they exist
                    exclude_columns = ['run_id', 'timestamp', 'id']
                    metric_columns = [col for col in display_metrics.index if col not in exclude_columns]
                    
                    # Create metrics table with all available metrics
                    metrics_dict = {
                        format_column_name(col): [format_metric_value(col, display_metrics[col])]
                        for col in metric_columns
                    }
                    
                    metrics_table = pd.DataFrame(metrics_dict)
                    st.table(metrics_table)
            
            # Add plot controls
            with st.expander("Plot Settings", expanded=False):
                # Date range selector
                min_date = pd.to_datetime(predictions_df['datetime'].min()).date()
                max_date = pd.to_datetime(predictions_df['datetime'].max()).date()
                
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date
                    )
                with col2:
                    end_date = st.date_input(
                        "End Date",
                        value=max_date,
                        min_value=min_date,
                        max_value=max_date
                    )
                
                # Filter data by date range
                mask = (predictions_df['datetime'].dt.date >= start_date) & (predictions_df['datetime'].dt.date <= end_date)
                filtered_df = predictions_df[mask].copy()
                
                if filtered_df.empty:
                    st.warning("No data available for the selected date range.")
                    return
                
                st.info(f"Showing data from {start_date} to {end_date} ({len(filtered_df):,} records)")
                
                # Calculate min and max points for the slider
                min_points = min(1000, len(filtered_df))
                max_points = min(20000, len(filtered_df))
                default_points = min(5000, len(filtered_df))
                
                if min_points == max_points:
                    st.info(f"Using {min_points:,} data points for plotting")
                    max_points = min_points
                else:
                    max_points = st.slider(
                        "Maximum points to plot",
                        min_value=min_points,
                        max_value=max_points,
                        value=default_points,
                        step=1000,
                        help="Higher values show more detail but may be slower to render"
                    )
            
            # Display predictions plots
            st.subheader(f"Prediction Analysis {page_info}")
            prediction_figs = analyzer.plot_predictions(filtered_df, max_points)
            for fig in prediction_figs:
                st.plotly_chart(fig, use_container_width=True)
            
            # Add download options in sidebar
            st.sidebar.subheader("Download Data")
            
            if view_option == "Paginated View":
                csv_predictions = filtered_df.to_csv(index=False)
                st.sidebar.download_button(
                    label="Download Filtered Data",
                    data=csv_predictions,
                    file_name=f"predictions_{selected_run_id}_filtered.csv",
                    mime="text/csv"
                )
            else:
                csv_predictions = filtered_df.to_csv(index=False)
                st.sidebar.download_button(
                    label="Download All Filtered Data",
                    data=csv_predictions,
                    file_name=f"predictions_{selected_run_id}_all_filtered.csv",
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

    # Model details sidebar
    if st.session_state.show_details_sidebar and details_col:
        with details_col:
            st.markdown("## Model Details")
            model_name = selected_run["model_name"]
            
            # Initialize ModelComparison to get model details
            comparison = ModelComparison(db_path)
            model_details = comparison.get_model_repository_details(model_name)
            
            if model_details:
                display_model_details_section(model_details)
            else:
                st.warning(f"No repository details found for {model_name}")


