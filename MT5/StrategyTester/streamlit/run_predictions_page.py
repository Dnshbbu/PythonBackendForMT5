import streamlit as st
import pandas as pd
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional
import mlflow
from run_predictions import HistoricalPredictor
import logging

def get_available_tables(db_path: str) -> List[str]:
    """Get list of available tables from the database, sorted by creation time (newest first)"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name, sql, tbl_name
            FROM sqlite_master 
            WHERE type='table' AND name LIKE 'strategy_%'
            ORDER BY name DESC;
        """)
        
        tables = [table[0] for table in cursor.fetchall()]
        conn.close()
        
        return tables
    except Exception as e:
        st.error(f"Error accessing database: {str(e)}")
        return []

def get_available_models(models_dir: str) -> List[str]:
    """Get list of available trained models, sorted by creation time (newest first)"""
    try:
        # Get all model files with their creation times
        model_files = []
        for file in os.listdir(models_dir):
            if file.endswith(('.joblib', '.pt', '.pth')) and not file.endswith('_scaler.joblib'):
                file_path = os.path.join(models_dir, file)
                # Get file creation time (or last modification time if creation time is not available)
                try:
                    creation_time = os.path.getctime(file_path)
                except OSError:
                    creation_time = os.path.getmtime(file_path)
                model_name = os.path.splitext(file)[0]
                model_files.append((model_name, creation_time))
        
        # Sort by creation time in descending order (newest first)
        model_files.sort(key=lambda x: x[1], reverse=True)
        
        # Extract only the model names
        sorted_models = [model[0] for model in model_files]
        
        # Log the order for verification
        if sorted_models:
            logging.info(f"Found {len(sorted_models)} models")
            logging.info(f"Most recent models: {sorted_models[:3]}")
        
        return sorted_models
    except Exception as e:
        st.error(f"Error accessing models directory: {str(e)}")
        return []

def get_equivalent_command(table: str, model_name: Optional[str], output_format: str, 
                         output_path: Optional[str], batch_size: int, force: bool) -> str:
    """Generate the equivalent command line command"""
    cmd_parts = ["python run_predictions.py"]
    cmd_parts.append(f"--table {table}")
    
    if model_name:
        cmd_parts.append(f"--model-name {model_name}")
    if output_format:
        cmd_parts.append(f"--output-format {output_format}")
    if output_path:
        cmd_parts.append(f"--output-path {output_path}")
    if batch_size != 1000:  # Only add if different from default
        cmd_parts.append(f"--batch-size {batch_size}")
    if force:
        cmd_parts.append("--force")
        
    return " ".join(cmd_parts)

def display_prediction_results(results_df: pd.DataFrame, model_name: str):
    """Display prediction results in a formatted way"""
    st.markdown("##### 📊 Prediction Results")
    
    # Key metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", len(results_df))
    with col2:
        st.metric("Mean Absolute Error", f"{results_df['Error'].abs().mean():.4f}")
    with col3:
        st.metric("RMSE", f"{(results_df['Error'] ** 2).mean() ** 0.5:.4f}")
    
    # Detailed results in tabs
    tabs = st.tabs(["Summary", "Detailed Results", "Visualization"])
    
    with tabs[0]:
        st.markdown("#### Summary Statistics")
        summary_stats = {
            "Model Used": model_name,
            "Data Points": len(results_df),
            "Time Range": f"{results_df.index.min()} to {results_df.index.max()}",
            "Mean Prediction": f"{results_df['Predicted_Price'].mean():.4f}",
            "Mean Actual": f"{results_df['Actual_Price'].mean():.4f}",
            "Mean Error": f"{results_df['Error'].mean():.4f}"
        }
        st.json(summary_stats)
    
    with tabs[1]:
        st.markdown("#### Detailed Results")
        st.dataframe(results_df)
        
        # Add download button
        csv = results_df.to_csv()
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with tabs[2]:
        st.markdown("#### Visualization")
        # Plot actual vs predicted
        st.line_chart(
            results_df[['Actual_Price', 'Predicted_Price']].rename(
                columns={'Actual_Price': 'Actual', 'Predicted_Price': 'Predicted'}
            )
        )
        # Plot error distribution
        st.markdown("#### Error Distribution")
        st.line_chart(results_df['Error'])

def run_predictions_page():
    """Streamlit interface for running predictions"""
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
    models_dir = os.path.join(current_dir, 'models')
    
    # Create left-right layout
    left_col, right_col = st.columns([1, 1], gap="large")
    
    # Left Column - Configuration
    with left_col:
        st.markdown("""
            <p style='color: #666; margin: 0; font-size: 0.9em;'>Configure prediction parameters</p>
            <hr style='margin: 0.2em 0 0.7em 0;'>
        """, unsafe_allow_html=True)
        
        # Table Selection
        st.markdown("##### 📊 Select Table")
        available_tables = get_available_tables(db_path)
        selected_table = st.selectbox(
            "Select Table for Predictions",
            options=available_tables,
            key="selected_table",
            help="Choose a table to run predictions on"
        )
        
        # Model Selection
        st.markdown("##### 🤖 Select Model")
        available_models = get_available_models(models_dir)
        model_name = st.selectbox(
            "Select Model",
            options=["Latest Model"] + available_models,
            key="model_name",
            help="Choose a model for predictions (default: latest model)"
        )
        if model_name == "Latest Model":
            model_name = None
        
        # Output Configuration
        st.markdown("##### 💾 Output Configuration")
        output_format = st.radio(
            "Output Format",
            options=['db', 'csv', 'both'],
            help="Choose where to save the predictions"
        )
        
        if output_format in ['csv', 'both']:
            output_path = st.text_input(
                "Output Path",
                value=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                help="Path for CSV output file"
            )
        else:
            output_path = None
        
        # Advanced Options
        with st.expander("⚙️ Advanced Options"):
            batch_size = st.number_input(
                "Batch Size",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                help="Number of records to process at once"
            )
            force_rerun = st.checkbox(
                "Force Rerun",
                help="Force rerun predictions even if they exist"
            )
        
        # Run Button
        st.markdown("##### 🚀 Run Predictions")
        run_button = st.button(
            "Run Predictions",
            type="primary",
            use_container_width=True,
            help="Click to start the prediction process"
        )
    
    # Right Column - Output
    with right_col:
        st.markdown("""
            <p style='color: #666; margin: 0; font-size: 0.9em;'>Prediction output and results</p>
            <hr style='margin: 0.2em 0 0.7em 0;'>
        """, unsafe_allow_html=True)
        
        # Command Preview
        if selected_table:
            st.markdown("##### 📝 Command Preview")
            cmd = get_equivalent_command(
                selected_table, model_name, output_format, 
                output_path, batch_size, force_rerun
            )
            st.code(cmd, language="bash")
        
        # Run predictions when button is clicked
        if run_button:
            if not selected_table:
                st.error("⚠️ Please select a table for predictions.")
            else:
                try:
                    st.markdown("##### 🔄 Execution")
                    st.info("Starting prediction process...")
                    
                    with st.spinner("Running predictions..."):
                        predictor = HistoricalPredictor(
                            db_path=db_path,
                            models_dir=models_dir,
                            model_name=model_name
                        )
                        
                        results_df = predictor.run_predictions(selected_table)
                        
                        # Handle output based on format
                        if output_format in ['csv', 'both']:
                            results_df.to_csv(output_path)
                            st.success(f"✅ Predictions saved to CSV: {output_path}")
                        
                        if output_format in ['db', 'both']:
                            st.success("✅ Predictions saved to database")
                        
                        # Display results
                        display_prediction_results(
                            results_df,
                            predictor.model_predictor.current_model_name
                        )
                        
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}") 