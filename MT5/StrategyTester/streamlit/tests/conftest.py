import pytest
import pandas as pd
import numpy as np
import os
import sqlite3
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import StandardScaler
import json
import xgboost as xgb

@pytest.fixture
def sample_timeseries_data():
    """Create sample time series data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='h')
    np.random.seed(42)
    
    data = {
        'Date': dates.date,
        'Time': dates.time,
        'Close': np.random.randn(len(dates)),
        'Volume': np.random.randint(1000, 10000, len(dates)),
        'Feature1': np.random.randn(len(dates)),
        'Feature2': np.random.randn(len(dates))
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def test_db_path(tmp_path, sample_timeseries_data):
    """Create a temporary SQLite database with sample data"""
    db_path = tmp_path / "test_trading.db"
    
    conn = sqlite3.connect(str(db_path))
    
    # Create tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS test_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Date DATE,
            Time TIME,
            Close FLOAT,
            Volume INTEGER,
            Feature1 FLOAT,
            Feature2 FLOAT
        )
    """)
    
    # Create model repository table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_repository (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT UNIQUE,
            model_type TEXT,
            training_type TEXT,
            prediction_horizon INTEGER,
            features TEXT,  -- JSON array of feature names
            feature_importance TEXT,  -- JSON object of feature importances
            model_params TEXT,  -- JSON object of model parameters
            metrics TEXT,  -- JSON object of model metrics
            training_tables TEXT,  -- JSON array of training table names
            training_period_start TIMESTAMP,
            training_period_end TIMESTAMP,
            data_points INTEGER,
            model_path TEXT,
            scaler_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            additional_metadata TEXT  -- JSON object for any additional metadata
        )
    """)
    
    # Insert sample data
    sample_timeseries_data.to_sql('test_table', conn, if_exists='replace', index=False)
    
    conn.close()
    
    return str(db_path)

@pytest.fixture
def test_models_dir(tmp_path):
    """Create a temporary directory for model storage"""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return str(models_dir)

@pytest.fixture
def feature_cols():
    """Define feature columns for testing"""
    return ['Close', 'Volume', 'Feature1', 'Feature2']

@pytest.fixture
def target_col():
    """Define target column for testing"""
    return 'Close'

@pytest.fixture
def sample_model(test_models_dir, feature_cols):
    """Create a sample XGBoost model for testing"""
    # Create a simple XGBoost model
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3)
    X = np.random.randn(100, len(feature_cols))
    y = np.random.randn(100)
    model.fit(X, y)
    
    # Save the model
    model_path = os.path.join(test_models_dir, "test_model.joblib")
    joblib.dump(model, model_path)
    
    # Create and save a scaler
    scaler = StandardScaler()
    scaler.fit(X)
    scaler_path = os.path.join(test_models_dir, "test_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    
    return {
        'model_name': 'test_model',
        'model_type': 'xgboost',
        'model_path': model_path,
        'scaler_path': scaler_path,
        'features': feature_cols
    }

@pytest.fixture
def setup_model_repository(test_db_path, sample_model):
    """Set up the model repository with a sample model"""
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()
    
    # Insert sample model into repository
    cursor.execute("""
        INSERT INTO model_repository 
        (model_name, model_type, training_type, prediction_horizon, features,
         feature_importance, model_params, metrics, training_tables,
         training_period_start, training_period_end, data_points,
         model_path, scaler_path, is_active, additional_metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        sample_model['model_name'],
        sample_model['model_type'],
        'batch',  # training_type
        1,  # prediction_horizon
        json.dumps(sample_model['features']),
        json.dumps({}),  # feature_importance
        json.dumps({'n_estimators': 10, 'max_depth': 3}),  # model_params
        json.dumps({'rmse': 0.1, 'r2': 0.8}),  # metrics
        json.dumps(['test_table']),  # training_tables
        datetime.now().isoformat(),  # training_period_start
        datetime.now().isoformat(),  # training_period_end
        100,  # data_points
        sample_model['model_path'],
        sample_model['scaler_path'],
        1,  # is_active
        json.dumps({})  # additional_metadata
    ))
    
    conn.commit()
    conn.close()
    
    return sample_model 