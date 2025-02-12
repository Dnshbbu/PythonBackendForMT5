import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import os
import sqlite3
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Conv1D, GlobalMaxPooling1D, Bidirectional, MultiHeadAttention, LayerNormalization
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

def create_gru_model(input_shape: tuple, layers: List[Dict], learning_rate: float = 0.001) -> Sequential:
    """Create a GRU model with specified architecture"""
    model = Sequential()
    
    # Add GRU layers
    for i, layer in enumerate(layers):
        return_sequences = i < len(layers) - 1  # Return sequences for all but last GRU layer
        if i == 0:
            model.add(GRU(
                units=layer['units'],
                return_sequences=return_sequences,
                input_shape=input_shape
            ))
        else:
            model.add(GRU(
                units=layer['units'],
                return_sequences=return_sequences
            ))
        
        if layer.get('dropout', 0) > 0:
            model.add(Dropout(layer['dropout']))
    
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

def create_cnn_lstm_model(input_shape: tuple, layers: List[Dict], learning_rate: float = 0.001) -> Sequential:
    """Create a CNN-LSTM model with specified architecture"""
    model = Sequential()
    
    # Add CNN layers for feature extraction
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.2))
    
    # Add LSTM layers
    for i, layer in enumerate(layers):
        return_sequences = i < len(layers) - 1
        model.add(LSTM(
            units=layer['units'],
            return_sequences=return_sequences
        ))
        if layer.get('dropout', 0) > 0:
            model.add(Dropout(layer['dropout']))
    
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

def create_bidirectional_lstm_model(input_shape: tuple, layers: List[Dict], learning_rate: float = 0.001) -> Sequential:
    """Create a Bidirectional LSTM model with specified architecture"""
    model = Sequential()
    
    # Add Bidirectional LSTM layers
    for i, layer in enumerate(layers):
        return_sequences = i < len(layers) - 1
        if i == 0:
            model.add(Bidirectional(
                LSTM(units=layer['units'], return_sequences=return_sequences),
                input_shape=input_shape
            ))
        else:
            model.add(Bidirectional(
                LSTM(units=layer['units'], return_sequences=return_sequences)
            ))
        
        if layer.get('dropout', 0) > 0:
            model.add(Dropout(layer['dropout']))
    
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

def create_transformer_model(input_shape: tuple, layers: List[Dict], learning_rate: float = 0.001) -> Sequential:
    """Create a Transformer model with specified architecture"""
    seq_len, d_model = input_shape
    inputs = tf.keras.Input(shape=input_shape)
    
    # Simple positional encoding with proper type handling
    positions = tf.cast(tf.range(seq_len), dtype=tf.float32)[:, tf.newaxis]
    div_term = tf.exp(
        tf.cast(tf.range(0, d_model, 2), dtype=tf.float32) * 
        (-tf.math.log(10000.0) / tf.cast(d_model, tf.float32))
    )
    
    # Create positional encoding matrix
    pos_encoding = tf.zeros((seq_len, d_model), dtype=tf.float32)
    
    # Calculate sin values for even indices
    sin_indices = tf.range(0, d_model, 2)
    sin_values = tf.sin(positions * div_term)
    pos_encoding = tf.tensor_scatter_nd_update(
        pos_encoding,
        tf.stack([
            tf.repeat(tf.range(seq_len), tf.shape(sin_indices)[0]),
            tf.tile(sin_indices, [seq_len])
        ], axis=1),
        tf.reshape(sin_values, [-1])
    )
    
    # Calculate cos values for odd indices
    cos_indices = tf.range(1, d_model, 2)
    if tf.shape(cos_indices)[0] > 0:  # Check if there are odd indices
        cos_values = tf.cos(positions * div_term[:tf.shape(cos_indices)[0]])
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            tf.stack([
                tf.repeat(tf.range(seq_len), tf.shape(cos_indices)[0]),
                tf.tile(cos_indices, [seq_len])
            ], axis=1),
            tf.reshape(cos_values, [-1])
        )
    
    # Add positional encoding to input
    x = tf.keras.layers.Add()([inputs, pos_encoding[tf.newaxis, ...]])
    
    # Add Transformer blocks
    for layer in layers:
        # Multi-head self attention
        attention_output = MultiHeadAttention(
            num_heads=layer.get('num_heads', 4),
            key_dim=layer.get('key_dim', 64)
        )(x, x, x)  # query, key, value
        
        # Add & Normalize (first residual connection)
        x = tf.keras.layers.Add()([x, attention_output])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ffn = tf.keras.Sequential([
            Dense(layer['units'] * 4, activation='relu'),
            Dense(d_model),  # Match input dimension
            Dropout(layer.get('dropout', 0))
        ])
        ffn_output = ffn(x)
        
        # Add & Normalize (second residual connection)
        x = tf.keras.layers.Add()([x, ffn_output])
        x = LayerNormalization(epsilon=1e-6)(x)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Final dense layer
    outputs = Dense(1)(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_tcn_model(input_shape: tuple, layers: List[Dict], learning_rate: float = 0.001) -> Sequential:
    """Create a Temporal Convolutional Network model"""
    model = Sequential()
    
    # Add TCN layers (using dilated convolutions)
    for i, layer in enumerate(layers):
        dilation_rate = 2 ** i  # Exponentially increasing dilation
        if i == 0:
            model.add(Conv1D(
                filters=layer['units'],
                kernel_size=3,
                dilation_rate=dilation_rate,
                padding='causal',
                activation='relu',
                input_shape=input_shape
            ))
        else:
            model.add(Conv1D(
                filters=layer['units'],
                kernel_size=3,
                dilation_rate=dilation_rate,
                padding='causal',
                activation='relu'
            ))
        
        if layer.get('dropout', 0) > 0:
            model.add(Dropout(layer['dropout']))
    
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

# Dictionary mapping model types to their creation functions
MODEL_CREATORS = {
    'lstm': create_lstm_model,
    'gru': create_gru_model,
    'cnn_lstm': create_cnn_lstm_model,
    'bidirectional_lstm': create_bidirectional_lstm_model,
    'transformer': create_transformer_model,
    'tcn': create_tcn_model
}

def load_and_validate_tables(db_path: str, selected_tables: List[str], target_col: str, feature_cols: List[str]) -> pd.DataFrame:
    """Load and validate multiple tables for training
    
    Args:
        db_path: Path to the SQLite database
        selected_tables: List of selected table names
        target_col: Name of the target column
        feature_cols: List of feature column names
        
    Returns:
        Combined DataFrame with data from all tables, sorted by DateTime
    """
    all_data = []
    
    try:
        conn = sqlite3.connect(db_path)
        
        for table_name in selected_tables:
            # Load data from table
            logging.info(f"Loading data from table: {table_name}")
            
            # Validate table has DateTime columns
            cursor = conn.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in cursor.fetchall()]
            if 'Date' not in columns or 'Time' not in columns:
                raise ValueError(f"Table {table_name} is missing required DateTime columns (Date, Time)")
            
            # Load data with proper ordering
            df = pd.read_sql_query(
                f"SELECT *, Date || ' ' || Time as DateTime FROM {table_name} ORDER BY Date, Time",
                conn
            )
            
            # Convert DateTime to pandas datetime
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            
            # Validate other required columns
            missing_cols = [col for col in [target_col] + feature_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Table {table_name} is missing required columns: {missing_cols}")
            
            # Add table identifier column
            df['source_table'] = table_name
            all_data.append(df)
            
            logging.info(f"Loaded {len(df)} rows from {table_name}")
            logging.info(f"Time range: {df['DateTime'].min()} to {df['DateTime'].max()}")
        
        conn.close()
        
        if not all_data:
            raise ValueError("No data was loaded from the selected tables")
        
        # Combine all data and sort by DateTime
        combined_df = pd.concat(all_data, axis=0, ignore_index=True)
        combined_df = combined_df.sort_values('DateTime')
        
        # Log temporal information
        logging.info(f"Combined dataset has {len(combined_df)} rows from {len(selected_tables)} tables")
        logging.info(f"Total time range: {combined_df['DateTime'].min()} to {combined_df['DateTime'].max()}")
        
        # Check for overlapping time periods
        for table_name in selected_tables:
            table_data = combined_df[combined_df['source_table'] == table_name]
            logging.info(f"Table {table_name} time range: {table_data['DateTime'].min()} to {table_data['DateTime'].max()}")
        
        return combined_df
        
    except Exception as e:
        if 'conn' in locals():
            conn.close()
        raise ValueError(f"Error loading tables: {str(e)}")

def prepare_sequences_multi_table(data: pd.DataFrame, sequence_length: int, target_col: str, 
                                feature_cols: List[str], n_lags: int = 3) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Prepare sequences for LSTM training with multiple tables
    
    This version ensures sequences don't cross table boundaries and maintains temporal consistency
    
    Args:
        data: Input DataFrame with features and target, must be sorted by DateTime
        sequence_length: Number of time steps to use as input sequence
        target_col: Name of the price column to predict
        feature_cols: List of feature column names to use for prediction
        n_lags: Number of previous price values to include as features
        
    Returns:
        X: Array of input sequences
        y: Array of target values
        scaler: Fitted MinMaxScaler
    """
    # Create a copy to avoid modifying original data
    df = data.copy()
    
    # Verify temporal ordering
    if not df['DateTime'].equals(df['DateTime'].sort_values()):
        raise ValueError("Data must be sorted by DateTime before preparing sequences")
    
    # Add lagged price values as features
    for i in range(1, n_lags + 1):
        lag_col = f"{target_col}_lag_{i}"
        # Calculate lags within each table group, respecting time order
        df[lag_col] = df.groupby('source_table')[target_col].shift(i)
        feature_cols.append(lag_col)
    
    # Add current price as a feature
    feature_cols.append(target_col)
    
    # Remove rows with NaN values from lagging
    df = df.dropna()
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols + [target_col]])
    
    X, y = [], []
    # Process each table separately while maintaining temporal order
    for table_name in df['source_table'].unique():
        table_mask = df['source_table'] == table_name
        table_data = scaled_data[table_mask]
        table_dates = df.loc[table_mask, 'DateTime']
        
        # Create sequences only within this table's data
        for i in range(len(table_data) - sequence_length):
            # Verify temporal continuity
            date_sequence = table_dates.iloc[i:i + sequence_length + 1]
            time_diffs = date_sequence.diff().dropna()
            
            # Check if the sequence is continuous (no large time gaps)
            if all(time_diff.total_seconds() <= 3600 for time_diff in time_diffs):  # 1 hour threshold
                X.append(table_data[i:(i + sequence_length), :-1])
                y.append(table_data[i + sequence_length, -1])
            else:
                logging.debug(f"Skipping sequence in {table_name} due to time gap at {date_sequence.iloc[0]}")
    
    if not X:
        raise ValueError("No valid sequences could be created after applying temporal consistency checks")
    
    logging.info(f"Created {len(X)} valid sequences after temporal consistency checks")
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
            
            # Model type selection
            model_type = st.selectbox(
                "Select Model Architecture",
                options=list(MODEL_CREATORS.keys()),
                format_func=lambda x: x.replace('_', ' ').upper(),
                help="""
                Available models:
                - LSTM: Long Short-Term Memory (standard)
                - GRU: Gated Recurrent Unit (faster alternative to LSTM)
                - CNN-LSTM: Combines CNN for feature extraction with LSTM
                - Bidirectional LSTM: Processes sequences in both directions
                - Transformer: Modern architecture with attention mechanism
                - TCN: Temporal Convolutional Network
                """
            )

            # Sequence length
            sequence_length = st.number_input(
                "Sequence Length",
                min_value=5,
                max_value=100,
                value=20,
                help="Number of time steps to use for prediction"
            )
            
            # Model-specific parameters
            if model_type == 'cnn_lstm':
                st.markdown("**CNN Parameters:**")
                cnn_filters = st.number_input("CNN Filters", 32, 256, 64)
                cnn_kernel_size = st.number_input("CNN Kernel Size", 2, 5, 3)
                
                st.markdown("**LSTM Layers:**")
            elif model_type == 'transformer':
                st.markdown("**Transformer Parameters:**")
                num_heads = st.number_input("Number of Attention Heads", 1, 8, 4)
                key_dim = st.number_input("Key Dimension", 16, 128, 64)
                
                st.markdown("**Feed-Forward Layers:**")
            elif model_type == 'tcn':
                st.markdown("**TCN Parameters:**")
                kernel_size = st.number_input("Kernel Size", 2, 5, 3)
                dilation_base = st.number_input("Dilation Base", 2, 4, 2)
                
                st.markdown("**Network Layers:**")
            else:
                st.markdown(f"**{model_type.upper()} Layers:**")
            
            num_layers = st.number_input(
                "Number of Layers", 
                1, 5, 2,
                help="Number of layers in the model"
            )
            
            layers = []
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
                    layer_config = {
                        'units': units,
                        'dropout': dropout
                    }
                    
                    # Add model-specific parameters
                    if model_type == 'transformer':
                        layer_config.update({
                            'num_heads': num_heads,
                            'key_dim': key_dim
                        })
                    elif model_type == 'cnn_lstm' and i == 0:
                        layer_config.update({
                            'filters': cnn_filters,
                            'kernel_size': cnn_kernel_size
                        })
                    elif model_type == 'tcn':
                        layer_config.update({
                            'kernel_size': kernel_size,
                            'dilation_base': dilation_base
                        })
                    
                    layers.append(layer_config)
            
            # Training parameters
            st.markdown("**Training Parameters:**")
            batch_size = st.number_input("Batch Size", 16, 256, 32)
            epochs = st.number_input("Epochs", 10, 1000, 100)
            learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
            
            # Model name - auto-generated based on model type and timestamp
            model_name = generate_model_name(
                model_type=model_type,
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
                    layers,
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
                                # Load and validate data from all selected tables
                                logging.info(f"Loading data from {len(st.session_state['dl_selected_tables'])} tables")
                                combined_df = load_and_validate_tables(
                                    db_path,
                                    st.session_state['dl_selected_tables'],
                                    target_col,
                                    selected_features
                                )
                                
                                # Check for stop
                                if check_dl_stop_clicked():
                                    raise DLTrainingInterrupt("Training stopped by user")
                                
                                # Prepare sequences with multi-table support
                                logging.info("Preparing sequences...")
                                X, y, scaler = prepare_sequences_multi_table(
                                    combined_df,
                                    sequence_length,
                                    target_col,
                                    selected_features.copy(),
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
                                logging.info(f"Creating {model_type.upper()} model...")
                                model_creator = MODEL_CREATORS[model_type]
                                model = model_creator(
                                    input_shape=(sequence_length, total_features),
                                    layers=layers,
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
                                        'layers': layers,
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