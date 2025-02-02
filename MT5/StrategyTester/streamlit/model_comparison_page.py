import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import logging
import json
from typing import List, Dict, Optional
from model_repository import ModelRepository

class ModelComparison:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.model_repository = ModelRepository(db_path)
        
    def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of unique models with their metadata"""
        query = """
            SELECT DISTINCT 
                model_name,
                source_table,
                COUNT(DISTINCT run_id) as total_runs,
                MIN(timestamp) as first_run,
                MAX(timestamp) as last_run
            FROM historical_prediction_metrics
            GROUP BY model_name, source_table
            ORDER BY last_run DESC
        """
        
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn)
            return df.to_dict('records')
        except Exception as e:
            logging.error(f"Error getting available models: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def get_model_metrics(self, model_names: List[str]) -> pd.DataFrame:
        """Get aggregated metrics for selected models"""
        placeholders = ','.join(['?' for _ in model_names])
        query = f"""
            SELECT 
                model_name,
                source_table,
                COUNT(DISTINCT run_id) as total_runs,
                AVG(mean_absolute_error) as avg_mae,
                AVG(root_mean_squared_error) as avg_rmse,
                AVG(mean_absolute_percentage_error) as avg_mape,
                AVG(r_squared) as avg_r2,
                AVG(direction_accuracy) as avg_direction_accuracy,
                AVG(up_prediction_accuracy) as avg_up_accuracy,
                AVG(down_prediction_accuracy) as avg_down_accuracy,
                AVG(price_volatility) as avg_volatility,
                AVG(error_skewness) as avg_error_skewness,
                AVG(first_quarter_accuracy) as avg_first_quarter_accuracy,
                AVG(last_quarter_accuracy) as avg_last_quarter_accuracy,
                AVG(max_correct_streak) as avg_max_streak,
                AVG(avg_correct_streak) as avg_streak
            FROM historical_prediction_metrics
            WHERE model_name IN ({placeholders})
            GROUP BY model_name, source_table
        """
        
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn, params=model_names)
            return df
        except Exception as e:
            logging.error(f"Error getting model metrics: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()

    def plot_metrics_comparison(self, metrics_df: pd.DataFrame) -> go.Figure:
        """Create enhanced comparison plots for model metrics"""
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                '<b>Error Metrics</b>', '<b>Directional Accuracy (%)</b>',
                '<b>R² Score</b>', '<b>Accuracy Score (%)</b>',
                '<b>Time Period Performance</b>', '<b>Volatility Analysis</b>',
                '<b>Error Skewness</b>', ''
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )

        # Enhanced color palette with better contrast
        colors = ['#00BFB3', '#FFA07A', '#7CB9E8']
        
        for idx, model in enumerate(metrics_df['model_name'].unique()):
            model_data = metrics_df[metrics_df['model_name'] == model]
            display_name = model.split('_')[-1]
            color = colors[idx % len(colors)]

            # Error Metrics - Improved scale and presentation
            error_metrics = ['MAE', 'RMSE', 'MAPE']
            error_values = [
                model_data['avg_mae'].iloc[0],
                model_data['avg_rmse'].iloc[0],
                model_data['avg_mape'].iloc[0]
            ]
            
            fig.add_trace(
                go.Bar(
                    name=display_name,
                    x=error_metrics,
                    y=error_values,
                    marker_color=color,
                    marker=dict(
                        line=dict(width=1, color='#333333')
                    ),
                    text=[f"{v:.4f}" for v in error_values],
                    textposition='auto',
                    textfont=dict(size=11)
                ),
                row=1, col=1
            )

            # R² Score - Horizontal bars for better readability
            fig.add_trace(
                go.Bar(
                    name=display_name,
                    x=[model_data['avg_r2'].iloc[0]],
                    y=[display_name],
                    orientation='h',
                    marker_color=color,
                    marker=dict(
                        line=dict(width=1, color='#333333')
                    ),
                    text=[f"{model_data['avg_r2'].iloc[0]:.4f}"],
                    textposition='auto',
                    textfont=dict(size=11)
                ),
                row=2, col=1
            )

            # Accuracy Score - Horizontal bars for consistency
            fig.add_trace(
                go.Bar(
                    name=display_name,
                    x=[model_data['avg_direction_accuracy'].iloc[0] * 100],
                    y=[display_name],
                    orientation='h',
                    marker_color=color,
                    marker=dict(
                        line=dict(width=1, color='#333333')
                    ),
                    text=[f"{model_data['avg_direction_accuracy'].iloc[0] * 100:.2f}%"],
                    textposition='auto',
                    textfont=dict(size=11)
                ),
                row=2, col=2
            )

            # Directional Accuracy with improved formatting
            fig.add_trace(
                go.Bar(
                    name=display_name,
                    x=['Overall', 'Upward', 'Downward'],
                    y=[
                        model_data['avg_direction_accuracy'].iloc[0] * 100,
                        model_data['avg_up_accuracy'].iloc[0] * 100,
                        model_data['avg_down_accuracy'].iloc[0] * 100
                    ],
                    marker_color=color,
                    marker=dict(
                        line=dict(width=1, color='#333333')
                    ),
                    text=[f"{v:.1f}%" for v in [
                        model_data['avg_direction_accuracy'].iloc[0] * 100,
                        model_data['avg_up_accuracy'].iloc[0] * 100,
                        model_data['avg_down_accuracy'].iloc[0] * 100
                    ]],
                    textposition='auto',
                ),
                row=1, col=2
            )

            # Streak Analysis
            fig.add_trace(
                go.Bar(
                    name=display_name,
                    x=['Max Streak', 'Avg Streak'],
                    y=[
                        model_data['avg_max_streak'].iloc[0],
                        model_data['avg_streak'].iloc[0]
                    ],
                    marker_color=color,
                    text=[
                        f"{model_data['avg_max_streak'].iloc[0]:.1f}",
                        f"{model_data['avg_streak'].iloc[0]:.1f}"
                    ],
                    textposition='auto',
                ),
                row=3, col=1
            )

            # Time Period Performance
            fig.add_trace(
                go.Bar(
                    name=display_name,
                    x=['First Quarter', 'Last Quarter'],
                    y=[
                        model_data['avg_first_quarter_accuracy'].iloc[0],
                        model_data['avg_last_quarter_accuracy'].iloc[0]
                    ],
                    marker_color=color,
                    text=[
                        f"{model_data['avg_first_quarter_accuracy'].iloc[0]:.4f}",
                        f"{model_data['avg_last_quarter_accuracy'].iloc[0]:.4f}"
                    ],
                    textposition='auto',
                ),
                row=3, col=1
            )

            # Volatility Analysis
            fig.add_trace(
                go.Bar(
                    name=display_name,
                    x=['Volatility'],
                    y=[model_data['avg_volatility'].iloc[0]],
                    marker_color=color,
                    text=[f"{model_data['avg_volatility'].iloc[0]:.4f}"],
                    textposition='auto',
                ),
                row=3, col=2
            )

            # Error Skewness (separate subplot)
            fig.add_trace(
                go.Bar(
                    name=display_name,
                    x=['Error Skewness'],
                    y=[model_data['avg_error_skewness'].iloc[0]],
                    marker_color=color,
                    text=[f"{model_data['avg_error_skewness'].iloc[0]:.4f}"],
                    textposition='auto',
                ),
                row=4, col=1
            )

        # Enhanced layout
        fig.update_layout(
            height=1200,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='#333333',
                borderwidth=1
            ),
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="#ffffff"
            )
        )

        # Specific subplot styling
        fig.update_xaxes(title=dict(text="Error Value", font=dict(size=11)), row=1, col=1)
        fig.update_xaxes(title=dict(text="R² Score", font=dict(size=11)), row=2, col=1)
        fig.update_xaxes(title=dict(text="Accuracy (%)", font=dict(size=11)), row=2, col=2)
        
        # Update ranges for better visualization
        fig.update_yaxes(range=[-0.5, len(metrics_df['model_name'].unique())-0.5], row=2, col=1)
        fig.update_yaxes(range=[-0.5, len(metrics_df['model_name'].unique())-0.5], row=2, col=2)
        
        # Add gridlines and improve axis appearance
        for row in range(1, 5):
            for col in [1, 2]:
                if row == 4 and col == 2:
                    continue
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    showline=True,
                    linewidth=1,
                    linecolor='#333333',
                    row=row,
                    col=col
                )
                fig.update_yaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    showline=True,
                    linewidth=1,
                    linecolor='#333333',
                    row=row,
                    col=col
                )

        return fig

    def get_model_repository_details(self, model_name: str) -> Dict:
        """Get detailed model information from the repository"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Debug logging
            logging.info(f"Searching for model details with name: {model_name}")
            
            # Remove .joblib extension if present
            model_name_without_ext = model_name.replace('.joblib', '')
            
            # First try exact match without extension
            query = """
                SELECT *
                FROM model_repository
                WHERE model_name = ?
            """
            
            cursor.execute(query, (model_name_without_ext,))
            row = cursor.fetchone()
            
            # If no exact match, try matching the base name
            if not row:
                base_model_name = model_name_without_ext.split('_20')[0]  # Split before the timestamp
                logging.info(f"No exact match found, trying with base name: {base_model_name}")
                
                query = """
                    SELECT *
                    FROM model_repository
                    WHERE model_name LIKE ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                cursor.execute(query, (f"{base_model_name}%",))
                row = cursor.fetchone()
            
            if row:
                columns = [description[0] for description in cursor.description]
                model_details = dict(zip(columns, row))
                logging.info(f"Found model details: {model_details['model_name']}")
                
                # Parse JSON strings
                for key in ['features', 'feature_importance', 'model_params', 'metrics', 
                          'training_tables', 'additional_metadata']:
                    if model_details.get(key):
                        try:
                            model_details[key] = json.loads(model_details[key])
                        except json.JSONDecodeError as e:
                            logging.error(f"Error parsing JSON for {key}: {e}")
                            model_details[key] = None
                return model_details
            
            logging.warning(f"No model details found for {model_name}")
            return None
            
        except Exception as e:
            logging.error(f"Error getting model repository details: {e}")
            return None
        finally:
            if conn:
                conn.close()

def create_compact_model_card(model: Dict) -> str:
    """Create a compact HTML card for model details"""
    return f"""
        <div style="
            font-size: 0.8rem;
            color: #888;
            margin-top: 5px;
            ">
            <span style="color: #00ADB5;">Model:</span> {model['model_name']} | 
            <span style="color: #00ADB5;">Source:</span> {model['source_table']} | 
            <span style="color: #00ADB5;">Runs:</span> {model['total_runs']} | 
            <span style="color: #00ADB5;">Last Run:</span> {model['last_run'][:10]}
        </div>
    """

def display_model_details_section(model_details: Dict):
    """Display model details using Streamlit components with modern styling"""
    if not model_details:
        st.warning("No details available for this model.")
        return
    
    try:
        # Custom CSS for modern styling
        st.markdown("""
            <style>
            .stExpander {
                background-color: #1a1c23;
                border: 1px solid #2d2d2d;
                border-radius: 8px;
                margin-bottom: 10px;
            }
            .stTabs {
                background-color: transparent;
            }
            .info-container {
                background-color: #252830;
                padding: 10px;
                border-radius: 6px;
                border: 1px solid #2d2d2d;
                margin-bottom: 10px;
            }
            .info-label {
                color: #00ADB5;
                font-size: 12px;
                font-weight: 500;
            }
            .info-value {
                color: #e0e0e0;
                font-size: 14px;
                margin-top: 4px;
            }
            .features-container {
                color: #e0e0e0;
                font-size: 14px;
                line-height: 1.6;
            }
            .features-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                margin-top: 10px;
            }
            .feature-count {
                color: #00ADB5;
                font-size: 16px;
                font-weight: 500;
                margin-bottom: 15px;
            }
            .feature-text {
                color: #e0e0e0;
                font-family: 'Arial', sans-serif;
                font-size: 14px;
                padding: 2px 0;
            }
            </style>
        """, unsafe_allow_html=True)

        # Create expandable section for each model
        with st.expander(f"📊 Model: {model_details['model_name'].split('_')[-1]}", expanded=False):
            # Create tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["Basic Info", "Training", "Features", "Importance"])
            
            # Tab 1: Basic Information
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                        <div class="info-container">
                            <div class="info-label">Model Type</div>
                            <div class="info-value">{}</div>
                        </div>
                        <div class="info-container">
                            <div class="info-label">Training Type</div>
                            <div class="info-value">{}</div>
                        </div>
                    """.format(
                        model_details.get('model_type', 'xgboost'),
                        model_details.get('training_type', 'multi')
                    ), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                        <div class="info-container">
                            <div class="info-label">Prediction Horizon</div>
                            <div class="info-value">{}</div>
                        </div>
                        <div class="info-container">
                            <div class="info-label">Data Points</div>
                            <div class="info-value">{:,}</div>
                        </div>
                    """.format(
                        model_details.get('prediction_horizon', 5),
                        model_details.get('data_points', 0)
                    ), unsafe_allow_html=True)

            # Tab 2: Training Period
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                        <div class="info-container">
                            <div class="info-label">Start Date</div>
                            <div class="info-value">{}</div>
                        </div>
                    """.format(model_details.get('training_period_start', '')[:10]), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                        <div class="info-container">
                            <div class="info-label">End Date</div>
                            <div class="info-value">{}</div>
                        </div>
                    """.format(model_details.get('training_period_end', '')[:10]), unsafe_allow_html=True)

            # Tab 3: Features
            with tab3:
                if model_details.get('features'):
                    st.markdown(f"""
                        <div class="feature-count">
                            Total Features: {len(model_details['features'])}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display features with consistent styling
                    for feature in model_details['features']:
                        st.markdown(f"""
                            <div class="feature-text">
                                {feature}
                            </div>
                        """, unsafe_allow_html=True)

            # Tab 4: Feature Importance
            with tab4:
                if model_details.get('feature_importance'):
                    # Sort and get top features
                    sorted_features = dict(sorted(
                        model_details['feature_importance'].items(), 
                        key=lambda x: float(x[1]) if isinstance(x[1], (int, float, str)) else 0, 
                        reverse=False
                    )[:10])  # Show top 8 features
                    
                    # Reverse the order for display (highest at top)
                    values = list(sorted_features.values())
                    keys = list(sorted_features.keys())
                    
                    # Create bar chart with improved styling
                    fig = go.Figure()
                    
                    # Add bars with reversed order
                    fig.add_trace(go.Bar(
                        x=values,
                        y=keys[::-1],  # Reverse the order of feature names
                        orientation='h',
                        marker=dict(
                            color=['rgba(0, 173, 181, 0.8)'] * len(sorted_features),
                            line=dict(color='#00ADB5', width=1)
                        ),
                        hovertemplate='<b>%{y}</b><br>' +
                                    'Importance: %{x:.4f}<extra></extra>'
                    ))
                    
                    # Update layout with better styling
                    fig.update_layout(
                        template="plotly_dark",
                        height=350,
                        margin=dict(l=20, r=20, t=30, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(26,28,35,0.5)',
                        title=dict(
                            text='Feature Importance (Top 10)',
                            x=0.5,
                            y=0.95,
                            xanchor='center',
                            yanchor='top',
                            font=dict(size=16, color='#00ADB5')
                        ),
                        xaxis=dict(
                            title='Importance Score',
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128, 128, 128, 0.2)',
                            zeroline=False,
                            tickformat='.3f',
                            title_font=dict(size=12, color='#8b8c8e'),
                            tickfont=dict(size=10, color='#8b8c8e')
                        ),
                        yaxis=dict(
                            title=None,
                            showgrid=False,
                            zeroline=False,
                            tickfont=dict(size=11, color='#e0e0e0')
                        ),
                        hoverlabel=dict(
                            bgcolor='#252830',
                            font_size=12,
                            font_family="Arial"
                        ),
                        bargap=0.2
                    )
                    
                    # Add value labels on bars
                    fig.update_traces(
                        texttemplate='%{x:.3f}',
                        textposition='outside',
                        textfont=dict(size=10, color='#e0e0e0'),
                        cliponaxis=False
                    )
                    
                    # Add subtle grid lines
                    fig.update_xaxes(showline=True, linewidth=1, linecolor='rgba(128, 128, 128, 0.2)')
                    fig.update_yaxes(showline=True, linewidth=1, linecolor='rgba(128, 128, 128, 0.2)')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                        <div style='color: #8b8c8e; font-size: 12px; margin-top: 10px;'>
                            * Higher values indicate greater importance in the model's predictions
                        </div>
                    """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error displaying model details: {str(e)}")
        logging.error(f"Error in display_model_details_section: {str(e)}")

def model_comparison_page():
    # Add additional CSS for sidebars and tooltip
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
        .streamlit-expanderHeader {
            background-color: #262730;
            color: #ffffff;
            border-radius: 10px;
        }
        .stDataFrame {
            background-color: #262730;
            border-radius: 10px;
        }
        .model-info {
            margin-top: 10px;
            padding: 10px;
            background-color: #1E1E1E;
            border-radius: 5px;
            border: 1px solid #333;
        }
        .metric-card {
            background-color: #1E1E1E;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #333;
            margin: 0.5rem 0;
        }
        .metric-value {
            color: #00ADB5;
            font-size: 1.5rem;
            font-weight: bold;
        }
        .metric-label {
            color: #888;
            font-size: 0.9rem;
        }
        .section-header {
            color: #00ADB5;
            font-size: 1.2rem;
            margin: 1rem 0;
            padding-bottom: 0.3rem;
            border-bottom: 2px solid #333;
        }
        .detail-section {
            background-color: #1E1E1E;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border: 1px solid #333;
        }
        .detail-section h4 {
            color: #00ADB5;
            margin-bottom: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header with info icon
    st.markdown("""
        <div class='header-container'>
            <h2 style='color: #00ADB5; padding: 1rem 0; margin: 0;'>
                Model Performance Comparison
            </h2>
            <div class='tooltip'>
                <span class='info-icon'>ℹ️</span>
                <div class='tooltiptext'>
                    <div style='margin-bottom: 10px;'>
                        We are comparing the models performance(metrics) from all previous runs.
                    </div>
                    <div style='margin-bottom: 10px;'>
                        This is sourced from "historical_prediction_metrics".
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

    # Left sidebar toggle button
    with st.sidebar:
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
        # Existing page content
        # st.markdown("""
        #     <h2 style='text-align: left; color: #00ADB5; padding: 1rem 0;'>
        #         Model Performance Comparison
        #     </h2>
        # """, unsafe_allow_html=True)

        # Initialize comparison
        db_path = "logs/trading_data.db"
        comparison = ModelComparison(db_path)

        # Get available models
        models = comparison.get_available_models()
        if not models:
            st.warning("No model metrics available in the database.")
            return

        # Model Selection Section
        st.markdown("<div class='section-header'>🤖 Select Models to Compare</div>", unsafe_allow_html=True)
        
        # Create a mapping of display names to full model info
        model_display_names = {
            f"{m['model_name'].split('_')[-1]} ({m['source_table']})": m['model_name']
            for m in models
        }

        # Multi-select for models
        selected_display_names = st.multiselect(
            "Choose models to compare:",
            options=list(model_display_names.keys()),
            default=[list(model_display_names.keys())[0]] if model_display_names else None,
            key="model_selector"
        )

        # Convert selected display names to model names
        selected_models = [model_display_names[display_name] for display_name in selected_display_names]

        # Show selected model details in an expander
        if selected_models:
            with st.expander("📋 Selected Models Details", expanded=False):
                for model in models:
                    if model['model_name'] in selected_models:
                        st.markdown(create_compact_model_card(model), unsafe_allow_html=True)

        if not selected_models:
            st.info("👆 Please select at least one model to analyze.")
            return

        try:
            # Get metrics for selected models
            metrics_df = comparison.get_model_metrics(selected_models)

            if metrics_df.empty:
                st.warning("No metrics data available for selected models.")
                return

            # Summary Cards in an expander
            with st.expander("📊 Summary Metrics", expanded=False):
                # Create metric cards in a grid
                metric_cols = st.columns(len(selected_models))
                for idx, model_name in enumerate(selected_models):
                    model_metrics = metrics_df[metrics_df['model_name'] == model_name].iloc[0]
                    
                    with metric_cols[idx]:
                        st.markdown(f"""
                            <div class='metric-card'>
                                <div class='metric-label'>Model</div>
                                <div class='metric-value'>{model_name.split('_')[-1]}</div>
                                <hr style='border-color: #333; margin: 0.5rem 0;'>
                                <div class='metric-label'>Direction Accuracy</div>
                                <div class='metric-value'>{model_metrics['avg_direction_accuracy']*100:.2f}%</div>
                                <div class='metric-label'>R² Score</div>
                                <div class='metric-value'>{model_metrics['avg_r2']:.4f}</div>
                                <div class='metric-label'>MAPE</div>
                                <div class='metric-value'>{model_metrics['avg_mape']:.2f}%</div>
                            </div>
                        """, unsafe_allow_html=True)

            # Detailed Metrics Table
            with st.expander("Detailed Metrics", expanded=True):
                detailed_metrics = metrics_df[[
                    'model_name', 'source_table', 'total_runs',
                    'avg_mae', 'avg_rmse', 'avg_mape', 'avg_r2',
                    'avg_direction_accuracy'
                ]].copy()
                
                detailed_metrics['avg_direction_accuracy'] = detailed_metrics['avg_direction_accuracy'] * 100
                detailed_metrics.columns = [
                    'Model Name', 'Data Source', 'Total Runs',
                    'Avg MAE', 'Avg RMSE', 'Avg MAPE (%)',
                    'Avg R²', 'Direction Accuracy (%)'
                ]
                
                st.dataframe(
                    detailed_metrics.style.format({
                        'Avg MAE': '{:.4f}',
                        'Avg RMSE': '{:.4f}',
                        'Avg MAPE (%)': '{:.2f}',
                        'Avg R²': '{:.4f}',
                        'Direction Accuracy (%)': '{:.2f}'
                    }).background_gradient(cmap='RdYlGn', subset=['Avg R²', 'Direction Accuracy (%)']),
                    use_container_width=True
                )

            # Visualization Section
            with st.expander("🎯 Performance Visualization", expanded=True):
                fig = comparison.plot_metrics_comparison(metrics_df)
                st.plotly_chart(fig, use_container_width=True)

            # Download Section
            with st.expander("💾 Export Data", expanded=False):
                csv = metrics_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Comparison Data (CSV)",
                    data=csv,
                    file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error comparing models: {str(e)}")
            logging.error(f"Error in model comparison page: {str(e)}")

    # Model details sidebar
    if st.session_state.show_details_sidebar and details_col:
        with details_col:
            st.markdown("## Model Details")
            for model_name in selected_models:
                model_details = comparison.get_model_repository_details(model_name)
                if model_details:
                    display_model_details_section(model_details)
                else:
                    st.warning(f"No repository details found for {model_name}")

if __name__ == "__main__":
    model_comparison_page() 