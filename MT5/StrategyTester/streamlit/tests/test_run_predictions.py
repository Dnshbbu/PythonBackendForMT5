import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sqlite3
from unittest.mock import Mock, patch
import joblib
import json
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from run_predictions import HistoricalPredictor
from model_predictor import ModelPredictor

@pytest.fixture
def sample_prediction_data():
    """Create sample prediction data"""
    return {
        'timestamp': datetime.now().isoformat(),
        'prediction': 100.5,
        'confidence': 0.85,
        'metrics': {
            'rmse': 0.1,
            'mae': 0.08,
            'direction_accuracy': 75.0
        }
    }

@pytest.fixture
def predictor(test_db_path, test_models_dir):
    """Create a HistoricalPredictor instance"""
    return HistoricalPredictor(
        db_path=test_db_path,
        models_dir=test_models_dir
    )

@pytest.fixture
def test_model(test_db_path, test_models_dir):
    """Create a test model for predictions"""
    # Create and train a simple model
    model = XGBRegressor(n_estimators=10)
    X = np.random.randn(100, 4)
    y = np.random.randn(100)
    model.fit(X, y)

    # Create and fit a scaler
    scaler = StandardScaler()
    scaler.fit(X)

    # Save model and scaler
    model_path = os.path.join(test_models_dir, "test_model.joblib")
    scaler_path = os.path.join(test_models_dir, "test_model_scaler.joblib")
    os.makedirs(test_models_dir, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # Insert model into database
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()
    
    # Drop the table if it exists and create it with the new schema
    cursor.execute("DROP TABLE IF EXISTS model_repository")
    cursor.execute("""
        CREATE TABLE model_repository (
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
    
    # Deactivate all existing models
    cursor.execute("UPDATE model_repository SET is_active = 0")
    
    # Insert or update the test model
    cursor.execute("""
        INSERT OR REPLACE INTO model_repository (
            model_name, model_type, training_type, prediction_horizon,
            features, feature_importance, model_params, metrics,
            training_tables, training_period_start, training_period_end,
            data_points, model_path, scaler_path, is_active, created_at,
            last_updated, additional_metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "test_model",
        "xgboost",
        "single",
        1,
        json.dumps(['Close', 'Volume', 'Feature1', 'Feature2']),
        json.dumps({}),  # Empty feature importance
        json.dumps({"n_estimators": 10}),  # Basic model params
        json.dumps({}),  # Empty metrics
        json.dumps(['test_table']),
        datetime.now().isoformat(),  # training_period_start
        datetime.now().isoformat(),  # training_period_end
        100,  # data_points
        model_path,
        scaler_path,
        1,  # is_active
        datetime.now().isoformat(),  # created_at
        datetime.now().isoformat(),  # last_updated
        None  # additional_metadata
    ))

    conn.commit()
    conn.close()

    return {
        'model_name': 'test_model',
        'model_type': 'xgboost',
        'features': ['Close', 'Volume', 'Feature1', 'Feature2'],
        'model_path': model_path,
        'scaler_path': scaler_path
    }

class TestHistoricalPredictor:
    def test_initialization(self, test_db_path, test_models_dir, test_model):
        """Test predictor initialization"""
        predictor = HistoricalPredictor(test_db_path, test_models_dir)
        assert predictor.db_path == test_db_path
        assert predictor.models_dir == test_models_dir

    def test_load_data(self, test_db_path, test_models_dir, test_model):
        """Test loading data from database"""
        predictor = HistoricalPredictor(test_db_path, test_models_dir)
        
        # Create test data with correct column names
        dates = pd.date_range(start='2023-01-01', periods=5)
        data = pd.DataFrame({
            'Date': dates.strftime('%Y-%m-%d'),
            'Time': dates.strftime('%H:%M:%S'),
            'Price': [1.0, 2.0, 1.5, 2.5, 2.0],
            'Volume': [100, 200, 150, 250, 200],
            'Feature1': [0.1, 0.2, 0.15, 0.25, 0.2],
            'Feature2': [0.5, 0.6, 0.55, 0.65, 0.6]
        })
        
        # Store test data in database
        with sqlite3.connect(test_db_path) as conn:
            data.to_sql('test_table', conn, if_exists='replace', index=False)
        
        loaded_data = predictor.load_data('test_table')
        assert len(loaded_data) > 0
        assert 'Price' in loaded_data.columns

    def test_store_predictions(self, test_db_path, test_models_dir, test_model):
        """Test storing predictions in database"""
        predictor = HistoricalPredictor(test_db_path, test_models_dir)
        
        # Create sample predictions DataFrame with the correct structure
        dates = pd.date_range(start='2023-01-01', periods=5)
        predictions = pd.DataFrame({
            'datetime': dates,
            'Actual_Price': np.random.randn(5),
            'Predicted_Price': np.random.randn(5),
            'Error': np.random.randn(5),
            'Price_Change': np.random.randn(5),
            'Predicted_Change': np.random.randn(5),
            'Price_Volatility': np.random.randn(5)
        })

        # Create sample summary metrics
        summary = {
            'total_predictions': 5,
            'mean_absolute_error': 0.1,
            'root_mean_squared_error': 0.15,
            'mean_absolute_percentage_error': 5.0,
            'r_squared': 0.85,
            'direction_accuracy': 75.0,
            'up_prediction_accuracy': 80.0,
            'down_prediction_accuracy': 70.0,
            'correct_ups': 3,
            'correct_downs': 2,
            'total_ups': 3,
            'total_downs': 2,
            'max_error': 0.5,
            'min_error': 0.01,
            'std_error': 0.12,
            'avg_price_change': 0.02,
            'price_volatility': 0.03,
            'mean_prediction_error': 0.08,
            'median_prediction_error': 0.07,
            'error_skewness': 0.1,
            'first_quarter_accuracy': 78.0,
            'last_quarter_accuracy': 72.0,
            'max_correct_streak': 8,
            'avg_correct_streak': 3.5
        }
        
        # Store predictions with summary
        predictor.store_predictions(predictions, summary, 'test_table', 'test_run_id')

    def test_store_metrics(self, test_db_path, test_models_dir, test_model):
        """Test storing metrics in database"""
        predictor = HistoricalPredictor(test_db_path, test_models_dir)
        
        # Create sample metrics with the correct structure
        metrics = {
            'total_predictions': 100,
            'mean_absolute_error': 0.1,
            'root_mean_squared_error': 0.15,
            'mean_absolute_percentage_error': 5.0,
            'r_squared': 0.85,
            'direction_accuracy': 75.0,
            'up_prediction_accuracy': 80.0,
            'down_prediction_accuracy': 70.0,
            'correct_ups': 40,
            'correct_downs': 35,
            'total_ups': 50,
            'total_downs': 50,
            'max_error': 0.5,
            'min_error': 0.01,
            'std_error': 0.12,
            'avg_price_change': 0.02,
            'price_volatility': 0.03,
            'mean_prediction_error': 0.08,
            'median_prediction_error': 0.07,
            'error_skewness': 0.1,
            'first_quarter_accuracy': 78.0,
            'last_quarter_accuracy': 72.0,
            'max_correct_streak': 8,
            'avg_correct_streak': 3.5
        }
        
        # Store metrics
        predictor.store_metrics(metrics, 'test_run_id', 'test_table')

    def test_run_predictions(self, test_db_path, test_models_dir, test_model):
        """Test running predictions"""
        predictor = HistoricalPredictor(test_db_path, test_models_dir)
        
        # Create test data with correct column names
        dates = pd.date_range(start='2023-01-01', periods=5)
        data = pd.DataFrame({
            'Date': dates.strftime('%Y-%m-%d'),
            'Time': dates.strftime('%H:%M:%S'),
            'Price': [1.0, 2.0, 1.5, 2.5, 2.0],
            'Volume': [100, 200, 150, 250, 200],
            'Feature1': [0.1, 0.2, 0.15, 0.25, 0.2],
            'Feature2': [0.5, 0.6, 0.55, 0.65, 0.6]
        })
        
        # Store test data in database
        with sqlite3.connect(test_db_path) as conn:
            data.to_sql('test_table', conn, if_exists='replace', index=False)
        
        result = predictor.run_predictions('test_table')
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_generate_summary(self, test_db_path, test_models_dir, test_model):
        """Test generating prediction summary"""
        predictor = HistoricalPredictor(test_db_path, test_models_dir)
        
        # Create sample data for summary generation
        dates = pd.date_range(start='2023-01-01', periods=5)
        data = pd.DataFrame({
            'datetime': dates,
            'Actual_Price': [1.0, 2.0, 1.5, 2.5, 2.0],
            'Predicted_Price': [1.1, 1.9, 1.6, 2.4, 2.1],
            'Error': [0.1, -0.1, 0.1, -0.1, 0.1],
            'Price_Change': [0, 1.0, -0.5, 1.0, -0.5],
            'Predicted_Change': [0, 0.8, -0.3, 0.8, -0.3],
            'Price_Volatility': [0.1, 0.2, 0.15, 0.25, 0.2]
        })
        
        summary = predictor.generate_summary(data)
        assert isinstance(summary, dict)

    def test_error_handling(self, test_db_path, test_models_dir, test_model):
        """Test error handling for invalid inputs"""
        predictor = HistoricalPredictor(test_db_path, test_models_dir)
        with pytest.raises(Exception):
            predictor.load_data('nonexistent_table')

    def test_calculate_streaks(self, test_db_path, test_models_dir, test_model):
        """Test streak calculations"""
        predictor = HistoricalPredictor(test_db_path, test_models_dir)
        
        # Create sample data for streak calculation
        correct_predictions = pd.Series([True, True, False, True, True, True, False, True])
        
        max_streak = predictor._calculate_max_streak(correct_predictions)
        avg_streak = predictor._calculate_avg_streak(correct_predictions)
        
        assert isinstance(max_streak, int)
        assert isinstance(avg_streak, float)
        assert max_streak == 3  # Longest streak is 3 (True, True, True)
        assert avg_streak == 2.0  # Average of streaks [2, 3, 1]