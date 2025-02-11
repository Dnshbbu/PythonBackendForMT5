import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict
import os
import sqlite3
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import logging
from datetime import datetime
from db_info import get_table_names, get_numeric_columns
from feature_config import get_feature_groups, get_all_features

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_lstm_model(input_shape: tuple, layers: List[Dict], learning_rate: float = 0.001) -> Sequential:
    """Create an LSTM model with specified architecture"""
    model = Sequential()
    
    # Add LSTM layers
    for i, layer in enumerate(layers):
        return_sequences = i < len(layers) - 1  # Return sequences for all but last LSTM layer
        if i == 0:
            model.add(LSTM(
                units=layer['units'],
                return_sequences=return_sequences,
                input_shape=input_shape
            ))
        else:
            model.add(LSTM(
                units=layer['units'],
                return_sequences=return_sequences
            ))
        
        if layer.get('dropout', 0) > 0:
            model.add(Dropout(layer['dropout']))
    
    # Add Dense output layer
    model.add(Dense(1))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def prepare_sequences(data: pd.DataFrame, sequence_length: int, target_col: str, feature_cols: List[str]):
    """Prepare sequences for LSTM training"""
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[feature_cols + [target_col]])
    
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length), :-1])
        y.append(scaled_data[i + sequence_length, -1])
    
    return np.array(X), np.array(y), scaler

def deep_learning_page():
    """Streamlit page for deep learning models"""
    st.markdown("""
        <h2 style='text-align: center;'>🧠 Deep Learning Models</h2>
        <p style='text-align: center; color: #666;'>
            Train deep learning models for time series prediction
        </p>
        <hr>
    """, unsafe_allow_html=True)
    
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
    models_dir = os.path.join(current_dir, 'models', 'deep_learning')
    os.makedirs(models_dir, exist_ok=True)
    
    # Create main left-right layout
    left_col, right_col = st.columns([1, 1], gap="large")
    
    # Left Column - Configuration
    with left_col:
        st.markdown("#### 📊 Data Configuration")
        
        # Table selection
        tables = get_table_names(db_path)
        selected_table = st.selectbox(
            "Select Table",
            options=tables,
            help="Choose the table to train on"
        )
        
        if selected_table:
            # Get columns
            numeric_cols = get_numeric_columns(db_path, selected_table)
            
            # Target selection
            target_col = st.selectbox(
                "Select Target Column",
                options=numeric_cols,
                index=numeric_cols.index('Price') if 'Price' in numeric_cols else 0
            )
            
            # Feature Selection
            st.markdown("#### 🎨 Feature Selection")
            feature_groups = get_feature_groups()
            all_features = get_all_features()
            
            selected_features = []
            for group_name, group_features in feature_groups.items():
                with st.expander(f"{group_name.title()} Features"):
                    for feature in group_features:
                        if feature in numeric_cols and st.checkbox(
                            feature,
                            value=True,
                            key=f"dl_feature_{feature}"
                        ):
                            selected_features.append(feature)
            
            # Model Configuration
            st.markdown("#### ⚙️ Model Configuration")
            
            # Sequence length
            sequence_length = st.number_input(
                "Sequence Length",
                min_value=5,
                max_value=100,
                value=20,
                help="Number of time steps to use for prediction"
            )
            
            # LSTM layers
            st.markdown("**LSTM Layers:**")
            num_layers = st.number_input("Number of LSTM Layers", 1, 5, 2)
            
            lstm_layers = []
            for i in range(num_layers):
                with st.expander(f"Layer {i+1}"):
                    units = st.number_input(
                        "Units",
                        min_value=8,
                        max_value=256,
                        value=64,
                        key=f"units_{i}"
                    )
                    dropout = st.slider(
                        "Dropout",
                        min_value=0.0,
                        max_value=0.5,
                        value=0.2,
                        key=f"dropout_{i}"
                    )
                    lstm_layers.append({
                        'units': units,
                        'dropout': dropout
                    })
            
            # Training parameters
            st.markdown("**Training Parameters:**")
            batch_size = st.number_input("Batch Size", 16, 256, 32)
            epochs = st.number_input("Epochs", 10, 1000, 100)
            learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
            
            # Model name
            model_name = st.text_input(
                "Model Name",
                value=f"lstm_model_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
            
            # Training button
            if st.button("🚀 Train Model", type="primary"):
                try:
                    with st.spinner("Training model..."):
                        # Load and prepare data
                        conn = sqlite3.connect(db_path)
                        df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
                        conn.close()
                        
                        # Prepare sequences
                        X, y, scaler = prepare_sequences(
                            df,
                            sequence_length,
                            target_col,
                            selected_features
                        )
                        
                        # Create and train model
                        model = create_lstm_model(
                            input_shape=(sequence_length, len(selected_features)),
                            layers=lstm_layers,
                            learning_rate=learning_rate
                        )
                        
                        # Train model
                        history = model.fit(
                            X, y,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.2,
                            verbose=1
                        )
                        
                        # Save model
                        model_path = os.path.join(models_dir, model_name)
                        os.makedirs(model_path, exist_ok=True)
                        model.save(os.path.join(model_path, 'model.h5'))
                        
                        # Save scaler and metadata
                        import joblib
                        joblib.dump(scaler, os.path.join(model_path, 'scaler.pkl'))
                        
                        metadata = {
                            'features': selected_features,
                            'target': target_col,
                            'sequence_length': sequence_length,
                            'layers': lstm_layers,
                            'training_params': {
                                'batch_size': batch_size,
                                'epochs': epochs,
                                'learning_rate': learning_rate
                            },
                            'training_history': {
                                'loss': history.history['loss'][-1],
                                'val_loss': history.history['val_loss'][-1],
                                'mae': history.history['mae'][-1],
                                'val_mae': history.history['val_mae'][-1]
                            }
                        }
                        
                        import json
                        with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
                            json.dump(metadata, f, indent=4)
                        
                        # Display results in right column
                        with right_col:
                            st.success(f"✨ Model trained successfully and saved to {model_path}")
                            
                            # Display training metrics
                            st.markdown("#### 📈 Training Results")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Final Loss", f"{history.history['loss'][-1]:.4f}")
                                st.metric("Final MAE", f"{history.history['mae'][-1]:.4f}")
                            with col2:
                                st.metric("Validation Loss", f"{history.history['val_loss'][-1]:.4f}")
                                st.metric("Validation MAE", f"{history.history['val_mae'][-1]:.4f}")
                            
                            # Plot training history
                            st.markdown("#### 📊 Training History")
                            history_df = pd.DataFrame(history.history)
                            st.line_chart(history_df)
                            
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    logging.error(f"Training error: {str(e)}", exc_info=True)
    
    # Right Column - Results
    with right_col:
        if not selected_table:
            st.info("👈 Please configure your model on the left to start training")

if __name__ == "__main__":
    setup_logging()
    deep_learning_page() 