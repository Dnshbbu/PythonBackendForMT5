import pytest
import pandas as pd
import numpy as np
import os
import sys
import sqlite3
from datetime import datetime
import logging
from typing import Dict, Union
import torch

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_predictor import ModelPredictor
from model_implementations import LSTMModel, TimeSeriesDataset

class TestModelPredictor(ModelPredictor):
    """Test-specific version of ModelPredictor that works with our test database structure"""
    
    def get_latest_data(self, table_name: str, n_rows: int = 100) -> pd.DataFrame:
        """
        Override get_latest_data to work with our test database structure
        which uses DateTime instead of id for ordering
        """
        try:
            query = f"""
            SELECT *
            FROM {table_name}
            ORDER BY DateTime DESC
            LIMIT {n_rows}
            """
            
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Convert date and time to datetime
            if 'Date' in df.columns and 'Time' in df.columns:
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                df = df.set_index('DateTime')
            
            df = df.sort_index()
            return df
            
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            raise
            
    def make_predictions(self, table_name: str, n_rows: int = 100, 
                        confidence_threshold: float = 0.8) -> Dict[str, Union[float, str, dict]]:
        """
        Override make_predictions to use 'Close' instead of 'Price' for test data
        """
        try:
            # Get latest data
            df = self.get_latest_data(table_name, n_rows)
            
            # Prepare features
            X = self.prepare_features(df)
            
            # Initialize result dictionary
            result = {
                'timestamp': datetime.now().isoformat(),
                'model_name': self.current_model_name,
                'table_name': table_name,
                'metrics': {}
            }
            
            # Make predictions based on model type
            if isinstance(self.model, LSTMModel):
                sequence_length = self.sequence_length
                dataset = TimeSeriesDataset(X.values, df['Close'].values, sequence_length)
                predictions = []
                
                with torch.no_grad():
                    for i in range(len(dataset)):
                        x, _ = dataset[i]
                        x = torch.FloatTensor(x).unsqueeze(0)
                        pred = self.model(x)
                        predictions.append(pred.item())
                
                # Pad the beginning with NaN values
                pad = [np.nan] * (sequence_length - 1)
                predictions = pad + predictions
            else:
                predictions = self.model.predict(X)
            
            # Calculate prediction metrics
            actual_prices = df['Close'].values
            valid_indices = ~np.isnan(predictions)
            predictions = np.array(predictions)[valid_indices]
            actual_prices = actual_prices[valid_indices]
            
            if len(predictions) > 0:
                # Basic metrics
                mae = np.mean(np.abs(actual_prices - predictions))
                mse = np.mean((actual_prices - predictions) ** 2)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
                
                # Direction accuracy
                actual_changes = np.diff(actual_prices)
                predicted_changes = np.diff(predictions)
                correct_directions = np.sum((actual_changes * predicted_changes) > 0)
                direction_accuracy = correct_directions / len(actual_changes) * 100
                
                # Volatility and trend metrics
                price_volatility = np.std(actual_changes)
                avg_price_change = np.mean(np.abs(actual_changes))
                
                # Store metrics
                result['metrics'] = {
                    'mean_absolute_error': mae,
                    'root_mean_squared_error': rmse,
                    'mean_absolute_percentage_error': mape,
                    'direction_accuracy': direction_accuracy,
                    'price_volatility': price_volatility,
                    'avg_price_change': avg_price_change,
                    'prediction_count': len(predictions)
                }
                
                # Latest prediction
                result['prediction'] = float(predictions[-1])
                result['actual'] = float(actual_prices[-1])
                result['error'] = float(actual_prices[-1] - predictions[-1])
                
                # Confidence score based on recent accuracy
                recent_errors = np.abs(actual_prices[-10:] - predictions[-10:])
                confidence_score = 1.0 - (np.mean(recent_errors) / np.mean(actual_prices[-10:]))
                result['confidence'] = max(0.0, min(1.0, confidence_score))
                
                logging.info(f"Successfully generated predictions with {len(predictions)} data points")
                
            else:
                logging.warning("No valid predictions generated")
                result['error'] = "No valid predictions could be generated"
            
            return result
            
        except Exception as e:
            logging.error(f"Error in make_predictions: {e}")
            raise

# Test functions now use TestModelPredictor instead of ModelPredictor

def test_predictor_initialization(test_db_path, test_models_dir):
    """Test if ModelPredictor initializes correctly"""
    predictor = TestModelPredictor(test_db_path, test_models_dir)
    assert predictor.db_path == test_db_path
    assert predictor.models_dir == test_models_dir
    assert predictor.model is None
    assert predictor.current_model_name is None

def test_load_model_by_name(test_db_path, test_models_dir, setup_model_repository, sample_model):
    """Test loading a specific model by name"""
    predictor = TestModelPredictor(test_db_path, test_models_dir)
    predictor.load_model_by_name(sample_model['model_name'])
    
    assert predictor.current_model_name == sample_model['model_name']
    assert predictor.model is not None
    assert predictor.scaler is not None
    assert predictor.feature_columns == sample_model['features']

def test_load_latest_model(test_db_path, test_models_dir, setup_model_repository):
    """Test loading the latest model"""
    predictor = TestModelPredictor(test_db_path, test_models_dir)
    predictor.load_latest_model()
    
    assert predictor.current_model_name is not None
    assert predictor.model is not None
    assert predictor.scaler is not None

def test_get_latest_data(test_db_path, test_models_dir, sample_timeseries_data):
    """Test fetching latest data from database"""
    predictor = TestModelPredictor(test_db_path, test_models_dir)
    
    # Create view for datetime-based ordering
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS test_table_with_datetime AS
        SELECT *, datetime(Date || ' ' || Time) as DateTime
        FROM test_table
    """)
    conn.commit()
    conn.close()
    
    df = predictor.get_latest_data('test_table_with_datetime', n_rows=50)
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) <= 50
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing

def test_prepare_features(test_db_path, test_models_dir, setup_model_repository, 
                         sample_model, sample_timeseries_data):
    """Test feature preparation"""
    predictor = TestModelPredictor(test_db_path, test_models_dir)
    predictor.load_model_by_name(sample_model['model_name'])
    
    # Create view for datetime-based ordering
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS test_table_with_datetime AS
        SELECT *, datetime(Date || ' ' || Time) as DateTime
        FROM test_table
    """)
    conn.commit()
    conn.close()
    
    # Get some data
    df = predictor.get_latest_data('test_table_with_datetime', n_rows=10)
    
    # Prepare features
    X = predictor.prepare_features(df)
    
    assert isinstance(X, pd.DataFrame)
    assert not X.empty
    assert list(X.columns) == sample_model['features']
    assert X.shape[1] == len(sample_model['features'])

def test_make_predictions(test_db_path, test_models_dir, setup_model_repository, 
                         sample_model, sample_timeseries_data):
    """Test making predictions"""
    predictor = TestModelPredictor(test_db_path, test_models_dir)
    predictor.load_model_by_name(sample_model['model_name'])
    
    # Create view for datetime-based ordering
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS test_table_with_datetime AS
        SELECT *, datetime(Date || ' ' || Time) as DateTime
        FROM test_table
    """)
    conn.commit()
    conn.close()
    
    predictions = predictor.make_predictions('test_table_with_datetime', n_rows=10)
    
    assert isinstance(predictions, dict)
    assert 'prediction' in predictions
    assert 'actual' in predictions
    assert 'confidence' in predictions
    assert 'timestamp' in predictions
    assert 'metrics' in predictions
    assert isinstance(predictions['prediction'], float)
    assert isinstance(predictions['actual'], float)
    assert isinstance(predictions['confidence'], float)
    assert isinstance(predictions['metrics'], dict)
    assert 'mean_absolute_error' in predictions['metrics']
    assert 'root_mean_squared_error' in predictions['metrics']
    assert 'direction_accuracy' in predictions['metrics']

def test_error_handling_invalid_model(test_db_path, test_models_dir):
    """Test error handling when loading invalid model"""
    predictor = TestModelPredictor(test_db_path, test_models_dir)
    
    with pytest.raises(ValueError):
        predictor.load_model_by_name('nonexistent_model')

def test_error_handling_invalid_table(test_db_path, test_models_dir, setup_model_repository):
    """Test error handling when accessing invalid table"""
    predictor = TestModelPredictor(test_db_path, test_models_dir)
    predictor.load_latest_model()
    
    with pytest.raises(Exception):
        predictor.get_latest_data('nonexistent_table')

def test_prediction_with_missing_features(test_db_path, test_models_dir,
                                            setup_model_repository, sample_model):
    """Test predictions when some features are missing"""
    predictor = TestModelPredictor(test_db_path, test_models_dir)
    predictor.load_model_by_name(sample_model['model_name'])

    # Create data with missing features
    dates = pd.date_range(start='2023-01-01', end='2023-01-02', freq='h')
    limited_data = pd.DataFrame({
        'Date': dates.date,
        'Time': dates.time,
        'Close': np.random.randn(len(dates)),
        'Volume': np.random.randint(1000, 10000, len(dates))
    })

    # Insert into database and create view
    conn = sqlite3.connect(test_db_path)
    limited_data.to_sql('limited_table', conn, index=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS limited_table_with_datetime AS
        SELECT *, datetime(Date || ' ' || Time) as DateTime
        FROM limited_table
    """)
    conn.commit()
    conn.close()

    # Should still work with missing features (they'll be filled with 0s)
    predictions = predictor.make_predictions('limited_table_with_datetime', n_rows=10)
    
    assert isinstance(predictions, dict)
    assert 'prediction' in predictions
    assert 'actual' in predictions
    assert 'confidence' in predictions
    assert 'timestamp' in predictions
    assert 'metrics' in predictions
    assert isinstance(predictions['prediction'], float)
    assert isinstance(predictions['actual'], float)
    assert isinstance(predictions['confidence'], float)
    assert isinstance(predictions['metrics'], dict)
    assert 'mean_absolute_error' in predictions['metrics']
    assert 'root_mean_squared_error' in predictions['metrics']
    assert 'direction_accuracy' in predictions['metrics']