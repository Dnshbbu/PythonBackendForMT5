import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Database settings
DATABASE_CONFIG = {
    'db_name': 'trading_data.db',
    'logs_dir': str(PROJECT_ROOT / 'logs'),
    'models_dir': str(PROJECT_ROOT / 'models'),
}

# Model training settings
MODEL_CONFIG = {
    'min_rows_for_training': 20,
    'batch_size': 10,
    'prediction_confidence_threshold': 0.8
}

# ZMQ Server settings
ZMQ_CONFIG = {
    'receive_address': "tcp://127.0.0.1:5556",
    'send_address': "tcp://127.0.0.1:5557",
    'signal_address': "tcp://127.0.0.1:5558"
}

# Ensure required directories exist
def create_required_directories():
    for dir_path in [DATABASE_CONFIG['logs_dir'], DATABASE_CONFIG['models_dir']]:
        os.makedirs(dir_path, exist_ok=True)

create_required_directories()