import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Optional

class ModelComparison:
    def __init__(self, db_path: str):
        self.db_path = db_path
        
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

def model_comparison_page():
    # Add custom CSS with enhanced styling
    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
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
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <h2 style='text-align: center; color: #00ADB5; padding: 1rem 0;'>
            Model Performance Comparison
        </h2>
    """, unsafe_allow_html=True)

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
        with st.expander("📈 Detailed Metrics", expanded=True):
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

if __name__ == "__main__":
    model_comparison_page() 