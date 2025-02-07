import pytest
import pandas as pd
import numpy as np
import os
import sys
import sqlite3
from datetime import datetime, timedelta

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_trainer import TimeSeriesModelTrainer
from model_implementations import ModelFactory

@pytest.fixture
def trainer(test_db_path, test_models_dir):
    """Create a TimeSeriesModelTrainer instance"""
    return TimeSeriesModelTrainer(test_db_path, test_models_dir)

class TestTimeSeriesModelTrainer:
    def test_initialization(self, test_db_path, test_models_dir):
        """Test trainer initialization"""
        trainer = TimeSeriesModelTrainer(test_db_path, test_models_dir)
        assert trainer.db_path == test_db_path
        assert trainer.models_dir == test_models_dir
        assert os.path.exists(test_models_dir)

    def test_load_data_from_db(self, trainer, sample_timeseries_data):
        """Test data loading from database"""
        df = trainer.load_data_from_db('test_table')
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_monotonic_increasing
        assert all(col in df.columns for col in ['Close', 'Volume', 'Feature1', 'Feature2'])

    def test_prepare_features_target(self, trainer, sample_timeseries_data, feature_cols, target_col):
        """Test feature preparation and target creation"""
        df = trainer.load_data_from_db('test_table')
        
        X, y = trainer.prepare_features_target(
            df=df,
            target_col=target_col,
            feature_cols=feature_cols,
            prediction_horizon=1
        )
        
        # Check features
        assert isinstance(X, pd.DataFrame)
        assert not X.empty
        assert list(X.columns) == feature_cols
        assert X.isna().sum().sum() == 0  # No NaN values
        
        # Check target
        assert isinstance(y, pd.Series)
        assert len(y) == len(X)
        assert y.isna().sum() == 0  # No NaN values

    def test_time_series_split(self, trainer, sample_timeseries_data, feature_cols, target_col):
        """Test time series cross-validation split"""
        df = trainer.load_data_from_db('test_table')
        
        X, y = trainer.prepare_features_target(
            df=df,
            target_col=target_col,
            feature_cols=feature_cols
        )
        
        splits = trainer.time_series_split(X, y, n_splits=3)
        
        assert len(splits) == 3
        for train_idx, test_idx in splits:
            # Check split properties
            assert len(train_idx) >= len(test_idx)  # Train set should be larger
            assert max(train_idx) < min(test_idx)  # No data leakage
            assert len(set(train_idx) & set(test_idx)) == 0  # No overlap

    def test_model_training(self, trainer, sample_timeseries_data, feature_cols, target_col):
        """Test model training functionality"""
        df = trainer.load_data_from_db('test_table')
        
        X, y = trainer.prepare_features_target(
            df=df,
            target_col=target_col,
            feature_cols=feature_cols
        )
        
        # Test with different model types
        model_types = ['xgboost', 'decision_tree', 'random_forest']
        for model_type in model_types:
            model_params = {
                'model_type': model_type,
                'n_estimators': 10,  # Small number for testing
                'max_depth': 3
            }
            
            model, metrics = trainer.train_model(X, y, model_params)
            
            assert model is not None
            assert isinstance(metrics, dict)
            assert 'rmse' in metrics
            assert 'r2' in metrics
            assert metrics['r2'] >= -1.0 and metrics['r2'] <= 1.0

    def test_error_handling(self, trainer):
        """Test error handling in trainer"""
        # Test invalid table name
        with pytest.raises(Exception):
            trainer.load_data_from_db('nonexistent_table')
        
        # Test invalid feature columns
        df = pd.DataFrame({'A': [1, 2, 3]})
        with pytest.raises(Exception):
            trainer.prepare_features_target(df, 'B', ['C'])
        
        # Test invalid model type
        X = pd.DataFrame({'feature': [1, 2, 3]})
        y = pd.Series([1, 2, 3])
        with pytest.raises(Exception):
            trainer.train_model(X, y, {'model_type': 'invalid_model'})

    def test_train_and_save_multi_table(self, trainer, sample_timeseries_data, feature_cols, target_col):
        """Test training and saving model with multiple tables"""
        table_names = ['test_table']
        model_params = {
            'model_type': 'xgboost',
            'n_estimators': 10,
            'max_depth': 3
        }
        
        model_path, metrics = trainer.train_and_save_multi_table(
            table_names=table_names,
            target_col=target_col,
            feature_cols=feature_cols,
            model_params=model_params
        )
        
        assert os.path.exists(model_path)
        assert isinstance(metrics, dict)
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'training_tables' in metrics
        assert metrics['training_tables'] == table_names 