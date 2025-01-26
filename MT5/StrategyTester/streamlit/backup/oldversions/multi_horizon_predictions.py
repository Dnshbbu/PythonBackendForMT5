import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List
import numpy as np
from multi_horizon_predictor import MultiHorizonPredictor

def create_predictions_plot(data: pd.DataFrame, predictions: Dict[int, Dict]) -> go.Figure:
    """Create a plot showing actual data and predictions for different horizons"""
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add actual price line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Price'],
            name="Actual Price",
            line=dict(color='blue')
        ),
        secondary_y=False
    )
    
    # Add predictions as markers
    colors = ['red', 'green', 'purple']  # Different colors for different horizons
    for (horizon, pred), color in zip(predictions.items(), colors):
        if 'prediction' in pred:
            fig.add_trace(
                go.Scatter(
                    x=[data.index[-1]],
                    y=[pred['prediction']],
                    name=f"{horizon}-Step Prediction",
                    mode='markers',
                    marker=dict(
                        color=color,
                        size=12,
                        symbol='diamond'
                    )
                ),
                secondary_y=False
            )
    
    # Update layout
    fig.update_layout(
        title="Price Predictions Across Different Horizons",
        xaxis_title="Time",
        yaxis_title="Price",
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def create_confidence_plot(predictions: Dict[int, Dict]) -> go.Figure:
    """Create a bar plot showing confidence levels for different horizons"""
    horizons = []
    confidences = []
    colors = []
    
    for horizon, pred in sorted(predictions.items()):
        if 'confidence' in pred:
            horizons.append(f"{horizon}-Step")
            confidences.append(pred['confidence'] * 100)
            colors.append('green' if pred['is_confident'] else 'red')
    
    fig = go.Figure(data=[
        go.Bar(
            x=horizons,
            y=confidences,
            marker_color=colors,
            text=[f"{conf:.1f}%" for conf in confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence by Horizon",
        xaxis_title="Prediction Horizon",
        yaxis_title="Confidence (%)",
        yaxis_range=[0, 100]
    )
    
    return fig

def display_feature_importance(predictions: Dict[int, Dict]):
    """Display feature importance across different horizons"""
    st.subheader("Feature Importance Analysis")
    
    # Create columns for each horizon
    cols = st.columns(len(predictions))
    
    for (horizon, pred), col in zip(sorted(predictions.items()), cols):
        if 'top_features' in pred:
            with col:
                st.write(f"### {horizon}-Step Horizon")
                
                # Create feature importance plot
                features = list(pred['top_features'].keys())
                importances = list(pred['top_features'].values())
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=importances,
                        y=features,
                        orientation='h'
                    )
                ])
                
                fig.update_layout(
                    title="Top Features",
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)

def multi_horizon_predictions_page():
    """Main function for the multi-horizon predictions page"""
    st.title("Multi-Horizon Price Predictions")
    
    # Initialize predictor
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        models_dir = os.path.join(current_dir, 'models')
        
        predictor = MultiHorizonPredictor(db_path, models_dir)
        
        # Load models
        with st.spinner("Loading models..."):
            models = predictor.load_all_models()
            
        if not models:
            st.error("No models found. Please ensure models are trained for different horizons.")
            return
            
        # Display available horizons
        horizons = list(models.keys())
        st.success(f"Found models for {len(horizons)} prediction horizons: {sorted(horizons)}")
        
        # Strategy selection
        available_tables = []
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            available_tables = [row[0] for row in cursor.fetchall() if row[0].startswith('strategy_')]
            conn.close()
        except Exception as e:
            st.error(f"Error loading strategies: {str(e)}")
            return
            
        if not available_tables:
            st.error("No strategy data found in database.")
            return
            
        selected_strategy = st.selectbox(
            "Select Strategy",
            options=available_tables,
            format_func=lambda x: x.replace('strategy_', '')
        )
        
        # Number of rows for analysis
        n_rows = st.slider(
            "Number of data points to analyze",
            min_value=50,
            max_value=500,
            value=100,
            step=50
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05
        )
        
        if st.button("Generate Predictions"):
            with st.spinner("Generating predictions..."):
                try:
                    # Make predictions
                    predictions = predictor.make_all_predictions(
                        table_name=selected_strategy,
                        n_rows=n_rows,
                        confidence_threshold=confidence_threshold
                    )
                    
                    # Get the data used for predictions
                    conn = sqlite3.connect(db_path)
                    data = pd.read_sql_query(
                        f"SELECT * FROM {selected_strategy} ORDER BY id DESC LIMIT {n_rows}",
                        conn
                    )
                    conn.close()
                    
                    # Convert date and time to datetime
                    data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
                    data = data.set_index('DateTime').sort_index()
                    
                    # Display predictions plot
                    st.plotly_chart(
                        create_predictions_plot(data, predictions),
                        use_container_width=True
                    )
                    
                    # Display confidence plot
                    st.plotly_chart(
                        create_confidence_plot(predictions),
                        use_container_width=True
                    )
                    
                    # Display feature importance
                    display_feature_importance(predictions)
                    
                    # Detailed predictions table
                    st.subheader("Detailed Predictions")
                    predictions_df = pd.DataFrame({
                        f"{horizon}-Step": {
                            "Prediction": pred['prediction'],
                            "Confidence": f"{pred['confidence']*100:.1f}%",
                            "Status": "Confident" if pred['is_confident'] else "Low Confidence"
                        }
                        for horizon, pred in sorted(predictions.items())
                    }).T
                    
                    st.dataframe(predictions_df)
                    
                    # Download predictions
                    predictions_json = pd.DataFrame(predictions).to_json()
                    st.download_button(
                        label="Download Predictions",
                        data=predictions_json,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
                    
    except Exception as e:
        st.error(f"Error initializing predictor: {str(e)}")

if __name__ == "__main__":
    multi_horizon_predictions_page()