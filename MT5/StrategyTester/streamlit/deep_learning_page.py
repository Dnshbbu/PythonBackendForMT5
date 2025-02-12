import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
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

class DLTrainingInterrupt(Exception):
    """Custom exception for interrupting the deep learning training process"""
    pass

class DLStreamlitHandler(logging.Handler):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        if 'dl_training_logs' not in st.session_state:
            st.session_state['dl_training_logs'] = []
        self.logs = st.session_state['dl_training_logs']
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.logs.append(msg)
            
            # Keep only last 1000 messages
            if len(self.logs) > 1000:
                self.logs = self.logs[-1000:]
            
            # Update the placeholder with all logs
            log_text = "\n".join(self.logs)
            self.placeholder.code(log_text)
            
            # Force a Streamlit rerun to update the UI
            st.session_state['dl_training_logs'] = self.logs
            
            # Check for stop after each log message
            if check_dl_stop_clicked():
                raise DLTrainingInterrupt("Training stopped by user")
            
        except DLTrainingInterrupt:
            raise
        except Exception:
            self.handleError(record)

def check_dl_stop_clicked():
    """Check if stop button was clicked"""
    return st.session_state.get('dl_stop_clicked', False)

def on_dl_stop_click():
    """Callback for stop button click"""
    st.session_state['dl_stop_clicked'] = True
    st.session_state['dl_stop_message'] = "⚠️ Training was stopped by user"

def setup_dl_logging(placeholder):
    """Configure logging settings with Streamlit output"""
    # Create Streamlit handler
    streamlit_handler = DLStreamlitHandler(placeholder)
    streamlit_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    
    # Get root logger and add handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our custom handler
    root_logger.addHandler(streamlit_handler)
    
    return streamlit_handler

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def initialize_dl_session_state():
    """Initialize session state variables for deep learning page"""
    if 'dl_selected_tables' not in st.session_state:
        st.session_state['dl_selected_tables'] = []
    if 'dl_table_selections' not in st.session_state:
        st.session_state['dl_table_selections'] = {}
    if 'dl_table_data' not in st.session_state:
        st.session_state['dl_table_data'] = []
    if 'dl_training_logs' not in st.session_state:
        st.session_state['dl_training_logs'] = []
    if 'dl_stop_clicked' not in st.session_state:
        st.session_state['dl_stop_clicked'] = False
    if 'dl_stop_message' not in st.session_state:
        st.session_state['dl_stop_message'] = None

def get_dl_available_tables(db_path: str) -> List[Dict]:
    """Get list of available tables from the database with detailed information"""
    # Check if training was stopped
    if st.session_state.get('dl_stop_clicked', False):
        return []
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get tables with their creation time from sqlite_master
        cursor.execute("""
            SELECT name 
            FROM sqlite_master 
            WHERE type='table' AND name LIKE 'strategy_%'
            ORDER BY name DESC
        """)
        
        tables = []
        for (table_name,) in cursor.fetchall():
            try:
                # Get date range
                cursor.execute(f'SELECT MIN(Date || " " || Time), MAX(Date || " " || Time) FROM {table_name}')
                start_time, end_time = cursor.fetchone()
                
                # Get row count
                cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
                row_count = cursor.fetchone()[0]
                
                # Get unique symbols
                cursor.execute(f'SELECT DISTINCT Symbol FROM {table_name}')
                symbols = [row[0] for row in cursor.fetchall()]
                
                # Create a display name that includes key information
                display_name = f"{table_name} ({start_time} to {end_time}, {row_count} rows)"
                
                tables.append({
                    'name': table_name,
                    'date_range': f"{start_time} to {end_time}",
                    'total_rows': row_count,
                    'symbols': symbols,
                    'display_name': display_name
                })
                
            except Exception as e:
                logging.error(f"Error getting details for table {table_name}: {str(e)}")
                continue
        
        conn.close()
        return tables
    
    except Exception as e:
        if not st.session_state.get('dl_stop_clicked', False):
            st.error(f"Error accessing database: {str(e)}")
        return []

def on_dl_table_selection_change():
    """Callback to handle table selection changes"""
    edited_rows = st.session_state['dl_table_editor']['edited_rows']
    table_data = pd.DataFrame(st.session_state['dl_table_data'])
    
    for idx, changes in edited_rows.items():
        if '🔍 Select' in changes:
            table_name = table_data.iloc[idx]['Table Name']
            st.session_state['dl_table_selections'][table_name] = changes['🔍 Select']
    
    # Update selected tables list
    st.session_state['dl_selected_tables'] = [
        name for name, is_selected in st.session_state['dl_table_selections'].items() 
        if is_selected
    ]

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

def prepare_sequences(data: pd.DataFrame, sequence_length: int, target_col: str, feature_cols: List[str], n_lags: int = 3):
    """Prepare sequences for LSTM training to predict the next row's price
    
    Args:
        data: Input DataFrame with features and target
        sequence_length: Number of time steps to use as input sequence
        target_col: Name of the price column to predict
        feature_cols: List of feature column names to use for prediction
        n_lags: Number of previous price values to include as features (default: 3)
        
    Returns:
        X: Array of shape (n_sequences, sequence_length, n_features) containing input sequences
        y: Array of shape (n_sequences,) containing next row's prices (targets)
        scaler: Fitted MinMaxScaler for feature scaling
        
    Example:
        If sequence_length = 3, creates sequences like:
        Input sequence (X)                                                     Target (y)
        [Features & Price from row 0,1,2]   ->        Price from row 3
        [Features & Price from row 1,2,3]   ->        Price from row 4
    """
    # Create a copy to avoid modifying original data
    df = data.copy()
    
    # Add lagged price values as features
    for i in range(1, n_lags + 1):
        lag_col = f"{target_col}_lag_{i}"
        df[lag_col] = df[target_col].shift(i)
        feature_cols.append(lag_col)
    
    # Add current price as a feature
    feature_cols.append(target_col)
    
    # Remove rows with NaN values from lagging
    df = df.iloc[n_lags:]
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols + [target_col]])
    
    X, y = [], []
    for i in range(len(df) - sequence_length):
        # Take sequence_length rows of features as input
        X.append(scaled_data[i:(i + sequence_length), :-1])
        # Take the price from the next row after the sequence as target
        y.append(scaled_data[i + sequence_length, -1])
    
    return np.array(X), np.array(y), scaler

class DLProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback to display training progress in Streamlit"""
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = None
    
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        self.current_batch = 0
        logging.info(f"\nEpoch {self.current_epoch}/{self.total_epochs}")
    
    def on_train_batch_end(self, batch, logs=None):
        if self.total_batches is None and hasattr(self.model, 'train_data_adapter'):
            self.total_batches = len(self.model.train_data_adapter.get_dataset())
        
        self.current_batch = batch + 1
        if self.total_batches:
            progress = f"{self.current_batch}/{self.total_batches}"
            metrics = ' - '.join(f"{k}: {v:.4f}" for k, v in logs.items())
            logging.info(f"Batch {progress} - {metrics}")
    
    def on_epoch_end(self, epoch, logs=None):
        metrics = ' - '.join(f"{k}: {v:.4f}" for k, v in logs.items())
        logging.info(f"Epoch {self.current_epoch}/{self.total_epochs} - {metrics}")

def get_dl_equivalent_command(table_names: List[str], target_col: str, feature_cols: List[str], 
                         sequence_length: int, lstm_layers: List[Dict], batch_size: int,
                         epochs: int, learning_rate: float, model_name: str) -> str:
    """Generate the equivalent command line command for deep learning training"""
    base_cmd = "python train_deep_learning_models.py"
    tables_arg = f"--tables {' '.join(table_names)}"
    target_arg = f"--target {target_col}"
    features_arg = f"--features {' '.join(feature_cols)}"
    sequence_arg = f"--sequence-length {sequence_length}"
    
    # Format LSTM layers configuration
    layers_config = []
    for i, layer in enumerate(lstm_layers):
        layers_config.append(f"layer{i+1}:{layer['units']}:{layer['dropout']}")
    layers_arg = f"--lstm-layers {' '.join(layers_config)}"
    
    # Training parameters
    batch_arg = f"--batch-size {batch_size}"
    epochs_arg = f"--epochs {epochs}"
    lr_arg = f"--learning-rate {learning_rate}"
    model_arg = f"--model-name {model_name}"
    
    return f"{base_cmd} {tables_arg} {target_arg} {features_arg} {sequence_arg} {layers_arg} {batch_arg} {epochs_arg} {lr_arg} {model_arg}".strip()

def generate_model_name(model_type: str, training_type: str, timestamp: Optional[str] = None) -> str:
    """Generate consistent model name
    
    Args:
        model_type: Type of model (e.g., 'xgboost', 'decision_tree')
        training_type: Type of training ('single', 'multi', 'incremental', 'base')
        timestamp: Optional timestamp, will generate if None
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"dl-{model_type}_{training_type}_{timestamp}"

def deep_learning_page():
    """Streamlit page for deep learning models"""
    # Initialize session state
    initialize_dl_session_state()
    
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
    
    # Get available tables with detailed information
    available_tables = get_dl_available_tables(db_path)
    
    # Create main left-right layout
    left_col, right_col = st.columns([1, 1], gap="large")

    # Create a single container for the right column that persists
    right_container = right_col.container()

    # Right Column - Results and Visualization
    with right_container:
        st.markdown("""
            <p style='color: #666; margin: 0; font-size: 0.9em;'>Training Results and Model Performance</p>
            <hr style='margin: 0.2em 0 0.7em 0;'>
        """, unsafe_allow_html=True)
        
        # Create placeholders for different sections
        status_placeholder = st.empty()  # For info/warning messages
        training_progress = st.empty()  # For training logs
        stop_button_placeholder = st.empty()  # For stop button
        metrics_placeholder = st.empty()  # For metrics and results
        
        # Show stop message if exists
        if st.session_state.get('dl_stop_message'):
            status_placeholder.warning(st.session_state['dl_stop_message'])
        elif not st.session_state['dl_selected_tables']:
            status_placeholder.info("👈 Please select tables and configure your model on the left to start training.")

        # Show training logs if they exist
        if st.session_state.get('dl_training_logs'):
            with training_progress:
                st.markdown("##### 📝 Training Progress")
                st.code("\n".join(st.session_state['dl_training_logs']))

    # Left Column - Configuration
    with left_col:
        st.markdown("""
            <p style='color: #666; margin: 0; font-size: 0.9em;'>Configure your Deep Learning training parameters</p>
            <hr style='margin: 0.2em 0 0.7em 0;'>
        """, unsafe_allow_html=True)
        
        if not available_tables and not st.session_state.get('dl_stop_clicked', False):
            st.warning("No strategy tables found in the database.")
            return

        # Table Selection Section
        st.markdown("##### 📊 Select Tables for Training")
        
        # Create or use existing table data
        if not st.session_state['dl_table_data']:
            table_data = []
            for t in available_tables:
                # Use the stored selection state or default to False
                is_selected = st.session_state['dl_table_selections'].get(t['name'], False)
                table_data.append({
                    '🔍 Select': is_selected,
                    'Table Name': t['name'],
                    'Date Range': t['date_range'],
                    'Rows': t['total_rows'],
                    'Symbols': ', '.join(t['symbols'])
                })
            st.session_state['dl_table_data'] = table_data
        
        table_df = pd.DataFrame(st.session_state['dl_table_data'])
        
        # Display table information with checkboxes
        edited_df = st.data_editor(
            table_df,
            hide_index=True,
            column_config={
                '🔍 Select': st.column_config.CheckboxColumn(
                    "Select",
                    help="Select tables for training",
                    default=False
                )
            },
            key='dl_table_editor',
            on_change=on_dl_table_selection_change
        )

        # Model Configuration Section
        if st.session_state['dl_selected_tables']:
            # Get columns from first selected table
            first_table = st.session_state['dl_selected_tables'][0]
            numeric_cols = get_numeric_columns(db_path, first_table)
            
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
            
            # Model name - auto-generated based on model type and timestamp
            model_name = generate_model_name(
                model_type="lstm",
                training_type="single",
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            
            # Add this before the Training button
            # Display equivalent command
            if selected_features:
                st.markdown("##### 💻 Equivalent Command")
                cmd = get_dl_equivalent_command(
                    st.session_state['dl_selected_tables'],
                    target_col,
                    selected_features,
                    sequence_length,
                    lstm_layers,
                    batch_size,
                    epochs,
                    learning_rate,
                    model_name
                )
                st.code(cmd, language='bash')
                
                # Add configuration for lagged features before the training button
                st.markdown("#### 🕒 Price Features Configuration")
                use_price_features = st.checkbox("Use Current and Previous Prices as Features", value=True,
                                               help="Include current and previous price values to improve prediction")
                if use_price_features:
                    n_lags = st.number_input("Number of Previous Prices to Use", 
                                           min_value=1, max_value=10, value=3,
                                           help="Number of previous price values to include as features")
                else:
                    n_lags = 0
                
                # Training button
                if st.button("🚀 Train Model", type="primary"):
                    try:
                        with st.spinner("Training model..."):
                            # Setup logging with Streamlit output
                            streamlit_handler = setup_dl_logging(training_progress)
                            
                            # Show stop button
                            with stop_button_placeholder:
                                st.button("🛑 Stop Training", 
                                        on_click=on_dl_stop_click,
                                        key="dl_stop_training",
                                        help="Click to stop the training process",
                                        type="secondary")
                            
                            try:
                                # Load and prepare data
                                conn = sqlite3.connect(db_path)
                                df = pd.read_sql_query(f"SELECT * FROM {first_table}", conn)
                                conn.close()
                                
                                # Log progress
                                logging.info(f"Loaded data from {first_table} with {len(df)} rows")
                                
                                # Check for stop
                                if check_dl_stop_clicked():
                                    raise DLTrainingInterrupt("Training stopped by user")
                                
                                # Prepare sequences
                                logging.info("Preparing sequences...")
                                X, y, scaler = prepare_sequences(
                                    df,
                                    sequence_length,
                                    target_col,
                                    selected_features.copy(),  # Create a copy to avoid modifying original list
                                    n_lags=n_lags if use_price_features else 0
                                )
                                
                                # Log feature information
                                total_features = len(selected_features) + (n_lags if use_price_features else 0) + (1 if use_price_features else 0)
                                logging.info(f"Total number of features: {total_features}")
                                logging.info(f"Sequence shape: {X.shape}")
                                if use_price_features:
                                    logging.info(f"Using current price and {n_lags} previous prices as features")
                                
                                # Check for stop
                                if check_dl_stop_clicked():
                                    raise DLTrainingInterrupt("Training stopped by user")
                                
                                # Create and train model
                                logging.info("Creating LSTM model...")
                                model = create_lstm_model(
                                    input_shape=(sequence_length, total_features),
                                    layers=lstm_layers,
                                    learning_rate=learning_rate
                                )
                                
                                # Custom training callback to check for stop
                                class StopTrainingCallback(tf.keras.callbacks.Callback):
                                    def on_epoch_end(self, epoch, logs=None):
                                        if check_dl_stop_clicked():
                                            self.model.stop_training = True
                                            raise DLTrainingInterrupt("Training stopped by user")
                                
                                # Train model with progress logging
                                logging.info("Starting model training...")
                                callbacks = [
                                    StopTrainingCallback(),
                                    DLProgressCallback(total_epochs=epochs)
                                ]
                                
                                # Clear previous logs before starting new training
                                st.session_state['dl_training_logs'] = []
                                
                                history = model.fit(
                                    X, y,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    validation_split=0.2,
                                    verbose=0,  # Set to 0 as we'll handle progress display
                                    callbacks=callbacks
                                )
                                
                                # Save model and results
                                if not check_dl_stop_clicked():
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
                                            'loss': float(history.history['loss'][-1]),
                                            'val_loss': float(history.history['val_loss'][-1]),
                                            'mae': float(history.history['mae'][-1]),
                                            'val_mae': float(history.history['val_mae'][-1])
                                        }
                                    }
                                    
                                    import json
                                    with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
                                        json.dump(metadata, f, indent=4)
                                    
                                    # Display results in right column
                                    with metrics_placeholder:
                                        st.success(f"✨ Model trained successfully and saved to {model_path}")
                                        
                                        # Create tabs for different visualizations
                                        tab1, tab2 = st.tabs(["📈 Training Metrics", "📊 Training History"])
                                        
                                        # Training Metrics Tab
                                        with tab1:
                                            st.markdown("#### Training Results")
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.metric(
                                                    "Final Loss",
                                                    f"{history.history['loss'][-1]:.4f}",
                                                    help="Mean Squared Error loss on training data"
                                                )
                                                st.metric(
                                                    "Final MAE",
                                                    f"{history.history['mae'][-1]:.4f}",
                                                    help="Mean Absolute Error on training data"
                                                )
                                            with col2:
                                                st.metric(
                                                    "Validation Loss",
                                                    f"{history.history['val_loss'][-1]:.4f}",
                                                    help="Mean Squared Error loss on validation data"
                                                )
                                                st.metric(
                                                    "Validation MAE",
                                                    f"{history.history['val_mae'][-1]:.4f}",
                                                    help="Mean Absolute Error on validation data"
                                                )
                                        
                                        # Training History Tab
                                        with tab2:
                                            st.markdown("#### Training History")
                                            history_df = pd.DataFrame(history.history)
                                            
                                            # Plot loss curves
                                            st.markdown("##### Loss Curves")
                                            loss_df = history_df[['loss', 'val_loss']]
                                            loss_df.columns = ['Training Loss', 'Validation Loss']
                                            st.line_chart(loss_df)
                                            
                                            # Plot MAE curves
                                            st.markdown("##### MAE Curves")
                                            mae_df = history_df[['mae', 'val_mae']]
                                            mae_df.columns = ['Training MAE', 'Validation MAE']
                                            st.line_chart(mae_df)
                
                            finally:
                                # Clean up logging handler
                                root_logger = logging.getLogger()
                                root_logger.removeHandler(streamlit_handler)
                                # Ensure stop button is removed in all cases
                                stop_button_placeholder.empty()
                                
                    except DLTrainingInterrupt:
                        if 'dl_stop_message' in st.session_state and st.session_state['dl_stop_message']:
                            status_placeholder.warning(st.session_state['dl_stop_message'])
                        # Clean up any partial training artifacts if needed
                        if 'model_path' in locals():
                            try:
                                logging.info(f"Cleaning up partial training artifacts in {model_path}")
                                # Add cleanup code here if needed
                            except Exception as cleanup_error:
                                logging.error(f"Error during cleanup: {str(cleanup_error)}")
                                
                    except Exception as e:
                        status_placeholder.error(f"Error during training: {str(e)}")
                        logging.error(f"Training error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    setup_logging()
    deep_learning_page() 