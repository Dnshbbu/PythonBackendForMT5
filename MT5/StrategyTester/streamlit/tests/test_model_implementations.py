import pytest
import pandas as pd
import numpy as np
import os
import torch
from datetime import datetime

from model_implementations import (
    XGBoostTimeSeriesModel,
    DecisionTreeTimeSeriesModel,
    RandomForestTimeSeriesModel,
    LSTMTimeSeriesModel,
    TimeSeriesDataset,
    ModelFactory
)

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 100
    n_features = 4
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randn(n_samples))
    
    return X, y

@pytest.fixture
def lstm_data():
    """Create sample data specifically for LSTM testing"""
    np.random.seed(42)
    n_samples = 100
    n_features = 4
    sequence_length = 10
    
    # Create data as pandas DataFrame and Series
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randn(n_samples))
    
    dataset = TimeSeriesDataset(X, y, sequence_length)
    return dataset, X, y

class TestXGBoostModel:
    def test_initialization(self):
        """Test XGBoost model initialization"""
        model = XGBoostTimeSeriesModel()
        assert model.model_name == "xgboost"
        assert model.model is None
        assert model.supports_incremental_learning() is True
        
    def test_training(self, sample_data):
        """Test XGBoost model training"""
        X, y = sample_data
        model = XGBoostTimeSeriesModel()
        
        trained_model, metrics = model.train(X, y)
        assert trained_model is not None
        assert isinstance(metrics, dict)
        assert 'training_score' in metrics
        assert 'feature_importance' in metrics
        assert metrics['n_features'] == X.shape[1]
        assert metrics['n_samples'] == len(X)
        
    def test_prediction(self, sample_data):
        """Test XGBoost model prediction"""
        X, y = sample_data
        model = XGBoostTimeSeriesModel()
        model.train(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)
        
    def test_feature_importance(self, sample_data):
        """Test XGBoost feature importance"""
        X, y = sample_data
        model = XGBoostTimeSeriesModel()
        model.train(X, y)
        
        importance = model.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) == X.shape[1]
        assert all(isinstance(v, float) for v in importance.values())
        
    def test_partial_fit(self, sample_data):
        """Test XGBoost incremental learning"""
        X, y = sample_data
        model = XGBoostTimeSeriesModel()
        
        # Initial training
        model.train(X[:50], y[:50])
        initial_trees = model.model.n_estimators
        
        # Partial fit
        metrics = model.partial_fit(X[50:], y[50:], n_estimators=10)
        assert model.model.n_estimators > initial_trees
        assert isinstance(metrics, dict)
        assert 'training_score' in metrics
        
    def test_save_load(self, sample_data, tmp_path):
        """Test XGBoost model saving and loading"""
        X, y = sample_data
        model = XGBoostTimeSeriesModel()
        model.train(X, y)
        
        # Save
        save_path = model.save(str(tmp_path))
        assert os.path.exists(save_path)
        
        # Load
        new_model = XGBoostTimeSeriesModel()
        new_model.load(save_path)
        assert new_model.model is not None
        
        # Verify predictions match
        orig_pred = model.predict(X)
        new_pred = new_model.predict(X)
        np.testing.assert_array_almost_equal(orig_pred, new_pred)

class TestDecisionTreeModel:
    def test_initialization(self):
        """Test Decision Tree model initialization"""
        model = DecisionTreeTimeSeriesModel()
        assert model.model_name == "decision_tree"
        assert model.model is None
        assert model.supports_incremental_learning() is False
        
    def test_training(self, sample_data):
        """Test Decision Tree model training"""
        X, y = sample_data
        model = DecisionTreeTimeSeriesModel()
        
        trained_model, metrics = model.train(X, y)
        assert trained_model is not None
        assert isinstance(metrics, dict)
        assert 'training_score' in metrics
        
    def test_prediction(self, sample_data):
        """Test Decision Tree model prediction"""
        X, y = sample_data
        model = DecisionTreeTimeSeriesModel()
        model.train(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)
        
    def test_no_partial_fit(self, sample_data):
        """Test that partial_fit raises NotImplementedError"""
        X, y = sample_data
        model = DecisionTreeTimeSeriesModel()
        
        with pytest.raises(NotImplementedError):
            model.partial_fit(X, y)

class TestRandomForestModel:
    def test_initialization(self):
        """Test Random Forest model initialization"""
        model = RandomForestTimeSeriesModel()
        assert model.model_name == "random_forest"
        assert model.model is None
        assert model.supports_incremental_learning() is False
        
    def test_training(self, sample_data):
        """Test Random Forest model training"""
        X, y = sample_data
        model = RandomForestTimeSeriesModel()
        
        trained_model, metrics = model.train(X, y)
        assert trained_model is not None
        assert isinstance(metrics, dict)
        assert 'training_score' in metrics
        assert 'feature_importance' in metrics
        
    def test_prediction(self, sample_data):
        """Test Random Forest model prediction"""
        X, y = sample_data
        model = RandomForestTimeSeriesModel()
        model.train(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)

class TestLSTMModel:
    def test_initialization(self):
        """Test LSTM model initialization"""
        model = LSTMTimeSeriesModel()
        assert model.model_name == "lstm"
        assert model.model is None
        assert model.supports_incremental_learning() is False
        
    def test_training(self, lstm_data):
        """Test LSTM model training"""
        dataset, X, y = lstm_data
        model = LSTMTimeSeriesModel()
        
        trained_model, metrics = model.train(
            pd.DataFrame(X), 
            pd.Series(y),
            sequence_length=10,
            num_epochs=2  # Small number for testing
        )
        assert trained_model is not None
        assert isinstance(metrics, dict)
        assert 'training_loss' in metrics
        
    def test_prediction(self, lstm_data):
        """Test LSTM model prediction"""
        dataset, X, y = lstm_data
        model = LSTMTimeSeriesModel()
        model.train(
            pd.DataFrame(X), 
            pd.Series(y),
            sequence_length=10,
            num_epochs=2
        )
        
        predictions = model.predict(pd.DataFrame(X))
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)
        
    def test_save_load(self, lstm_data, tmp_path):
        """Test LSTM model saving and loading"""
        dataset, X, y = lstm_data
        model = LSTMTimeSeriesModel()
        model.train(
            pd.DataFrame(X), 
            pd.Series(y),
            sequence_length=10,
            num_epochs=2
        )
        
        # Save
        save_path = model.save(str(tmp_path))
        assert os.path.exists(save_path)
        
        # Load
        new_model = LSTMTimeSeriesModel()
        new_model.load(save_path)
        assert new_model.model is not None

class TestTimeSeriesDataset:
    def test_dataset_creation(self):
        """Test TimeSeriesDataset creation and access"""
        # Create sample data as numpy arrays
        X = np.random.randn(100, 4)
        y = np.random.randn(100)
        sequence_length = 10

        # Convert to pandas DataFrame/Series for testing
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
        y = pd.Series(y)

        dataset = TimeSeriesDataset(X, y, sequence_length)

        # Test dataset length
        assert len(dataset) == len(X) - sequence_length

        # Test getting an item
        x, y_true = dataset[0]
        assert isinstance(x, torch.FloatTensor)
        assert isinstance(y_true, torch.FloatTensor)
        assert x.shape == torch.Size([sequence_length, 4])  # [sequence_length, n_features]
        assert y_true.shape == torch.Size([])  # Single value

    def test_dataset_with_pandas(self):
        """Test TimeSeriesDataset with pandas DataFrame/Series"""
        # Create sample data as pandas DataFrame/Series
        X = pd.DataFrame(np.random.randn(100, 4), columns=['f1', 'f2', 'f3', 'f4'])
        y = pd.Series(np.random.randn(100))
        sequence_length = 10
        
        dataset = TimeSeriesDataset(X, y, sequence_length)
        assert len(dataset) == len(X) - sequence_length
        
        # Test getting an item
        x, y_true = dataset[0]
        assert isinstance(x, torch.FloatTensor)
        assert isinstance(y_true, torch.FloatTensor)
        assert x.shape == (sequence_length, X.shape[1])
        assert y_true.shape == torch.Size([])  # Single value

class TestModelFactory:
    def test_model_registration(self):
        """Test model registration and retrieval"""
        # Get available models
        available_models = ModelFactory.get_available_models()
        assert isinstance(available_models, list)
        assert len(available_models) > 0
        
        # Test getting each model type
        for model_type in available_models:
            model = ModelFactory.get_model(model_type)
            assert model is not None
            assert model.model_name == model_type
            
    def test_invalid_model_type(self):
        """Test getting invalid model type"""
        with pytest.raises(ValueError):
            ModelFactory.get_model("invalid_model_type") 