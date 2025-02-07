import pytest
import pandas as pd
import numpy as np
import os
import sys
import sqlite3
from datetime import datetime
import json

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_time_price_predictor import RealTimePricePredictor

@pytest.fixture
def sample_data_point():
    """Create a sample data point for testing"""
    return {
        'Price': '100.50',
        'Score': '0.75',
        'ExitScore': '0.25',
        'Factors': 'trend=up|strength=0.8|volatility=medium',
        'ExitFactors': 'profit=0.5|risk=0.2',
        'EntryScore': 'confidence=high|signal=buy'
    }

def test_initialization(test_db_path, test_models_dir, setup_model_repository):
    """Test if RealTimePricePredictor initializes correctly"""
    predictor = RealTimePricePredictor(test_db_path, test_models_dir)
    assert predictor.model_predictor is not None
    assert predictor.models_dir == test_models_dir
    assert len(predictor.selected_features) > 0
    assert predictor.data_buffer is not None
    assert predictor.data_buffer.maxlen == 2

def test_load_features_from_repository(test_db_path, test_models_dir, setup_model_repository):
    """Test loading features from model repository"""
    predictor = RealTimePricePredictor(test_db_path, test_models_dir)
    features = predictor.load_features_from_repository()
    assert isinstance(features, list)
    assert len(features) > 0
    assert all(isinstance(f, str) for f in features)

def test_load_model_by_name(test_db_path, test_models_dir, setup_model_repository, sample_model):
    """Test loading a specific model by name"""
    predictor = RealTimePricePredictor(test_db_path, test_models_dir)
    predictor.load_model_by_name(sample_model['model_name'])
    assert predictor.model_predictor.current_model_name == sample_model['model_name']
    assert predictor.model_predictor.model is not None

class TestRealTimePredictorNoModel(RealTimePricePredictor):
    """Test-specific version that doesn't require model loading"""
    def __init__(self, db_path: str, models_dir: str):
        self.model_predictor = None
        self.models_dir = models_dir
        self.setup_logging()
        self.data_buffer = None
        self.training_manager = None
        self.selected_features = ['Price', 'Score', 'ExitScore']

def test_split_factor_string(test_db_path, test_models_dir):
    """Test splitting factor strings into components"""
    predictor = TestRealTimePredictorNoModel(test_db_path, test_models_dir)
    
    # Test with numeric values
    result = predictor.split_factor_string('Test', 'value1=0.5|value2=1.0')
    assert result['Test_value1'] == 0.5
    assert result['Test_value2'] == 1.0
    
    # Test with boolean values
    result = predictor.split_factor_string('Test', 'flag1=true|flag2=false')
    assert result['Test_flag1'] == 1.0
    assert result['Test_flag2'] == 0.0
    
    # Test with empty string
    result = predictor.split_factor_string('Test', '')
    assert result == {}

def test_process_raw_data(test_db_path, test_models_dir, sample_data_point):
    """Test processing raw data points"""
    predictor = TestRealTimePredictorNoModel(test_db_path, test_models_dir)
    processed = predictor.process_raw_data(sample_data_point)
    
    # Check numeric fields
    assert isinstance(processed['Price'], float)
    assert isinstance(processed['Score'], float)
    assert isinstance(processed['ExitScore'], float)
    
    # Check processed factor fields
    assert 'Factors_trend' in processed
    assert 'ExitFactors_profit' in processed
    assert 'EntryScore_confidence' in processed
    
    # Check numeric conversion
    assert processed['Factors_strength'] == 0.8
    assert processed['ExitFactors_profit'] == 0.5

def test_prepare_features(test_db_path, test_models_dir, setup_model_repository, sample_data_point):
    """Test feature preparation"""
    predictor = RealTimePricePredictor(test_db_path, test_models_dir)
    processed_data = predictor.process_raw_data(sample_data_point)
    features = predictor.prepare_features(processed_data)
    
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    assert features.shape[0] == 1
    assert features.shape[1] == len(predictor.selected_features)
    assert all(features.dtypes == 'float64')

def test_make_prediction(test_db_path, test_models_dir, setup_model_repository, sample_data_point):
    """Test making predictions"""
    predictor = RealTimePricePredictor(test_db_path, test_models_dir)
    processed_data = predictor.process_raw_data(sample_data_point)
    features = predictor.prepare_features(processed_data)
    prediction = predictor.make_prediction(features)
    
    assert isinstance(prediction, dict)
    assert 'prediction' in prediction
    assert 'confidence' in prediction
    assert 'is_confident' in prediction
    assert 'top_features' in prediction
    assert 'timestamp' in prediction
    assert 'model_type' in prediction
    assert isinstance(prediction['prediction'], float)
    assert isinstance(prediction['confidence'], float)
    assert isinstance(prediction['is_confident'], bool)
    assert isinstance(prediction['top_features'], dict)
    assert len(prediction['top_features']) <= 5

def test_add_data_point(test_db_path, test_models_dir, setup_model_repository, sample_data_point):
    """Test adding data points and getting predictions"""
    predictor = RealTimePricePredictor(test_db_path, test_models_dir)
    
    # Add first data point
    result1 = predictor.add_data_point(sample_data_point)
    assert result1 is not None
    assert isinstance(result1, dict)
    assert len(predictor.data_buffer) == 1
    
    # Add second data point
    result2 = predictor.add_data_point(sample_data_point)
    assert result2 is not None
    assert isinstance(result2, dict)
    assert len(predictor.data_buffer) == 2
    
    # Add third data point (should maintain buffer size of 2)
    result3 = predictor.add_data_point(sample_data_point)
    assert result3 is not None
    assert isinstance(result3, dict)
    assert len(predictor.data_buffer) == 2

def test_error_handling(test_db_path, test_models_dir, setup_model_repository):
    """Test error handling for invalid inputs"""
    predictor = RealTimePricePredictor(test_db_path, test_models_dir)
    
    # Test with invalid model name
    with pytest.raises(Exception):
        predictor.load_model_by_name('nonexistent_model')
    
    # Test with None data point
    with pytest.raises(Exception):
        predictor.process_raw_data(None)
    
    # Test with empty feature data - should handle gracefully
    features = predictor.prepare_features({})
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    assert features.shape[1] == len(predictor.selected_features)
    assert all(features.dtypes == 'float64')
    assert all(features.iloc[0] == 0.0)  # All features should be filled with zeros
    
    # Test with None feature data - should handle gracefully
    features = predictor.prepare_features(None)
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    assert features.shape[1] == len(predictor.selected_features)
    assert all(features.dtypes == 'float64')
    assert all(features.iloc[0] == 0.0)  # All features should be filled with zeros

    # Test with list feature data - should handle gracefully
    features = predictor.prepare_features([1, 2, 3])
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    assert features.shape[1] == len(predictor.selected_features)
    assert all(features.dtypes == 'float64')
    assert all(features.iloc[0] == 0.0)  # All features should be filled with zeros

    # Test with invalid feature data type (string) - should handle gracefully
    features = predictor.prepare_features("invalid")
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    assert features.shape[1] == len(predictor.selected_features)
    assert all(features.dtypes == 'float64')
    assert all(features.iloc[0] == 0.0)  # All features should be filled with zeros 