from abc import ABC, abstractmethod
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
import logging
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
from incremental_learning import (
    IncrementalLearningMixin, 
    BatchedRetrainingMixin,
    ModelTrainingMetrics,
    ModelVersionManager
)
from model_base import BaseModel
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class BaseTimeSeriesModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.metrics_tracker = ModelTrainingMetrics()
        self.version_manager = ModelVersionManager()

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[Any, Dict]:
        """Train the model and return metrics"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        pass

    def save(self, save_dir: str, timestamp: Optional[str] = None) -> str:
        """Save model and metadata with proper type conversion"""
        try:
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
            model_path = os.path.join(save_dir, f"{self.model_name}_{timestamp}.joblib")
            joblib.dump(self.model, model_path)
            
            if self.scaler:
                scaler_path = os.path.join(save_dir, f"{self.model_name}_{timestamp}_scaler.joblib")
                joblib.dump(self.scaler, scaler_path)

            # Save feature names
            if self.feature_columns:
                feature_path = os.path.join(save_dir, f"{self.model_name}_{timestamp}_feature_names.json")
                with open(feature_path, 'w') as f:
                    json.dump({'feature_names': self.feature_columns}, f)

            # Save feature importance
            importance = self.get_feature_importance()
            converted_importance = {
                str(k): float(v) if hasattr(v, 'dtype') else float(v)
                for k, v in importance.items()
            }
            
            importance_path = os.path.join(save_dir, f"{self.model_name}_{timestamp}_feature_importance.json")
            with open(importance_path, 'w') as f:
                json.dump(converted_importance, f, indent=4)

            return model_path
            
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise

    def load(self, model_path: str) -> None:
        """Load model and associated metadata"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load model
            self.model = joblib.load(model_path)
            base_name = os.path.splitext(model_path)[0]
            
            # Load feature names
            feature_path = f"{base_name}_feature_names.json"
            if os.path.exists(feature_path):
                with open(feature_path, 'r') as f:
                    feature_data = json.load(f)
                self.feature_columns = feature_data['feature_names']
            
            # Load scaler if available
            scaler_path = f"{base_name}_scaler.joblib"
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

class XGBoostTimeSeriesModel(BaseTimeSeriesModel, IncrementalLearningMixin):
    def __init__(self):
        super().__init__("xgboost")
        self.default_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'reg:squarederror'
        }
        self.training_history = []

    def supports_incremental_learning(self) -> bool:
        return True

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[Any, Dict]:
        try:
            params = {**self.default_params, **kwargs}
            self.model = xgb.XGBRegressor(**params)
            
            # Store feature columns during training
            self.feature_columns = list(X.columns)
            
            # Train the model
            self.model.fit(X, y)
            
            # Calculate metrics
            score = float(self.model.score(X, y))
            feature_importance = self.get_feature_importance()
            
            metrics = {
                'training_score': score,
                'feature_importance': feature_importance,
                'n_features': len(self.feature_columns),
                'n_samples': len(X)
            }
            
            # Save feature names alongside model
            if hasattr(self, 'model_path'):
                self._save_feature_names()
            
            return self.model, metrics
            
        except Exception as e:
            logging.error(f"Error in XGBoost training: {e}")
            raise

    def _save_feature_names(self):
        """Save feature names to a JSON file"""
        if hasattr(self, 'feature_columns') and self.feature_columns:
            base_name = os.path.splitext(os.path.basename(self.model_path))[0]
            feature_path = os.path.join(
                os.path.dirname(self.model_path), 
                f"{base_name}_features.json"
            )
            with open(feature_path, 'w') as f:
                json.dump({
                    'feature_names': self.feature_columns,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=4)
            logging.info(f"Saved feature names to {feature_path}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        if self.model is None:
            return {}
            
        importance_dict = {}
        importances = self.model.feature_importances_
        feature_names = self.feature_columns or range(len(importances))
        
        for feat, imp in zip(feature_names, importances):
            importance_dict[str(feat)] = float(imp)
            
        return importance_dict

    def partial_fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        if self.model is None:
            return self.train(X, y, **kwargs)[1]
            
        n_new_trees = kwargs.get('n_estimators', 10)
        self.model.n_estimators += n_new_trees
        self.model.fit(X, y, xgb_model=self.model)
        
        return {
            'training_score': self.model.score(X, y),
            'feature_importance': self.get_feature_importance(),
            'n_trees': self.model.n_estimators
        }

    def needs_retraining(self, new_data_size: int) -> bool:
        """Determine if XGBoost model needs retraining"""
        if not self.model:
            logging.info("No existing model found, needs full training")
            return True
            
        # Check if new data would increase trees beyond limit
        current_trees = self.model.n_estimators
        max_trees = 2000  # Maximum number of trees to maintain model efficiency
        
        # Calculate estimated new trees needed
        estimated_new_trees = int(new_data_size / 100)  # One tree per 100 samples
        total_trees = current_trees + estimated_new_trees
        
        if total_trees > max_trees:
            logging.info(f"Would exceed max trees ({max_trees}), needs full retraining")
            return True
            
        # Check if new data is too large compared to original training set
        if self.training_history:
            last_training = self.training_history[-1]
            original_size = last_training.get('data_size', 0)
            if original_size > 0 and new_data_size / original_size > 0.5:
                logging.info("New data too large compared to original training set")
                return True
                
        return False

class DecisionTreeTimeSeriesModel(BaseTimeSeriesModel, BatchedRetrainingMixin, IncrementalLearningMixin):
    def __init__(self):
        super().__init__("decision_tree")
        BatchedRetrainingMixin.__init__(self)
        self.default_params = {
            'max_depth': 6,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }

    def supports_incremental_learning(self) -> bool:
        return False  # Decision Trees don't support true incremental learning

    def partial_fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """Not implemented for Decision Trees"""
        raise NotImplementedError("Decision Trees do not support incremental learning")

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[Any, Dict]:
        params = {**self.default_params, **kwargs}
        self.model = DecisionTreeRegressor(**params)
        self.model.fit(X, y)
        
        predictions = self.model.predict(X)
        metrics = {
            'training_score': self.model.score(X, y),
            'feature_importance': self.get_feature_importance()
        }
        
        self.metrics_tracker.add_metrics(metrics, 'full')
        self.record_training(metrics, len(X))
        return self.model, metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.model:
            return {}
        return dict(zip(self.feature_columns or [], 
                       self.model.feature_importances_))



class RandomForestTimeSeriesModel(BaseTimeSeriesModel, BatchedRetrainingMixin):
    def __init__(self):
        super().__init__("random_forest")
        BatchedRetrainingMixin.__init__(self)
        self.default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }

    def supports_incremental_learning(self) -> bool:
        return False  # Random Forests don't support true incremental learning

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[Any, Dict]:
        try:
            params = {**self.default_params, **kwargs}
            self.model = RandomForestRegressor(**params)
            
            # Store feature columns during training
            self.feature_columns = list(X.columns)
            
            # Train the model
            self.model.fit(X, y)
            
            # Calculate metrics
            score = float(self.model.score(X, y))
            predictions = self.model.predict(X)
            mse = np.mean((y - predictions) ** 2)
            rmse = np.sqrt(mse)
            
            metrics = {
                'training_score': score,
                'rmse': float(rmse),
                'feature_importance': self.get_feature_importance(),
                'n_features': len(self.feature_columns),
                'n_samples': len(X),
                'n_trees': self.model.n_estimators
            }
            
            self.metrics_tracker.add_metrics(metrics, 'full')
            self.record_training(metrics, len(X))
            return self.model, metrics
            
        except Exception as e:
            logging.error(f"Error in Random Forest training: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.model:
            return {}
            
        importance_dict = {}
        importances = self.model.feature_importances_
        feature_names = self.feature_columns or range(len(importances))
        
        for feat, imp in zip(feature_names, importances):
            importance_dict[str(feat)] = float(imp)
            
        return importance_dict

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length=10):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        return (self.X[idx:idx+self.sequence_length], 
                self.y[idx+self.sequence_length-1])

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class LSTMTimeSeriesModel(BaseTimeSeriesModel):
    def __init__(self):
        super().__init__("lstm")
        self.default_params = {
            'hidden_size': 64,
            'num_layers': 2,
            'sequence_length': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 10,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        self.scaler = StandardScaler()

    def supports_incremental_learning(self) -> bool:
        return False

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[Any, Dict]:
        try:
            params = {**self.default_params, **kwargs}
            self.feature_columns = list(X.columns)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Create dataset and dataloader
            dataset = TimeSeriesDataset(X_scaled, y, params['sequence_length'])
            dataloader = DataLoader(
                dataset, 
                batch_size=params['batch_size'],
                shuffle=False
            )
            
            # Initialize model
            self.model = LSTMModel(
                input_size=len(self.feature_columns),
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers']
            ).to(params['device'])
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=params['learning_rate'])
            
            # Training loop
            self.model.train()
            training_loss = []
            
            for epoch in range(params['num_epochs']):
                epoch_loss = 0
                for batch_X, batch_y in dataloader:
                    batch_X = batch_X.to(params['device'])
                    batch_y = batch_y.to(params['device'])
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y.unsqueeze(1))
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                training_loss.append(avg_loss)
                
                if (epoch + 1) % 10 == 0:
                    logging.info(f'Epoch [{epoch+1}/{params["num_epochs"]}], Loss: {avg_loss:.4f}')
            
            # Calculate final metrics using sequence-aware prediction
            self.model.eval()
            predictions = self.predict(X)
            
            # Adjust predictions and target lengths to match
            seq_length = params['sequence_length']
            valid_indices = slice(seq_length - 1, len(y))
            y_valid = y.iloc[valid_indices]
            predictions_valid = predictions[valid_indices]
            
            mse = np.mean((y_valid.values - predictions_valid) ** 2)
            rmse = np.sqrt(mse)
            
            metrics = {
                'training_loss': training_loss[-1],
                'rmse': float(rmse),
                'n_features': len(self.feature_columns),
                'n_samples': len(X),
                'sequence_length': params['sequence_length'],
                'hidden_size': params['hidden_size'],
                'num_layers': params['num_layers']
            }
            
            # Return self instead of self.model to maintain proper object for saving
            return self, metrics
            
        except Exception as e:
            logging.error(f"Error in LSTM training: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        
        # Create sequences for prediction
        sequence_length = self.default_params['sequence_length']
        predictions = np.zeros(len(X))
        
        # Handle the initial sequence - use first feature column instead of 'Price'
        first_feature = X.columns[0]
        for i in range(sequence_length - 1):
            predictions[i] = X[first_feature].iloc[i]  # Use first feature for initial sequence
        
        # Create dataset and dataloader for remaining predictions
        dataset = TimeSeriesDataset(
            pd.DataFrame(X_scaled, columns=X.columns), 
            pd.Series(np.zeros(len(X))),  # Dummy target
            sequence_length
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=self.default_params['batch_size'],
            shuffle=False
        )
        
        current_idx = sequence_length - 1
        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.default_params['device'])
                outputs = self.model(batch_X)
                batch_predictions = outputs.cpu().numpy().flatten()
                
                # Store predictions
                end_idx = min(current_idx + len(batch_predictions), len(predictions))
                predictions[current_idx:end_idx] = batch_predictions[:end_idx-current_idx]
                current_idx = end_idx
                
                if current_idx >= len(predictions):
                    break
        
        return predictions

    def get_feature_importance(self) -> Dict[str, float]:
        # LSTM doesn't provide direct feature importance
        # We'll return equal importance for all features
        if not self.feature_columns:
            return {}
        importance = 1.0 / len(self.feature_columns)
        return {feat: importance for feat in self.feature_columns}

    def save(self, save_dir: str, model_name: Optional[str] = None) -> str:
        """Save model and metadata"""
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"single_{timestamp}"
        
        # Add model type prefix here
        full_model_name = f"{self.model_name}_{model_name}"
        model_path = os.path.join(save_dir, f"{full_model_name}.pt")
        scaler_path = os.path.join(save_dir, f"{full_model_name}_scaler.joblib")
        
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_columns': self.feature_columns,
            'model_params': self.default_params
        }, model_path)
        
        # Save scaler
        joblib.dump(self.scaler, scaler_path)
        
        return model_path

    def load(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load model state and metadata
        checkpoint = torch.load(model_path)
        self.feature_columns = checkpoint['feature_columns']
        self.default_params = checkpoint['model_params']
        
        # Initialize and load model
        self.model = LSTMModel(
            input_size=len(self.feature_columns),
            hidden_size=self.default_params['hidden_size'],
            num_layers=self.default_params['num_layers']
        ).to(self.default_params['device'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load scaler
        scaler_path = model_path.replace('.pt', '_scaler.joblib')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

class ModelFactory:
    _models = {}

    @classmethod
    def register(cls, name: str, model_class: type):
        """Register a new model type"""
        if not issubclass(model_class, BaseTimeSeriesModel):
            raise ValueError("Model must inherit from BaseTimeSeriesModel")
        cls._models[name.lower()] = model_class

    @classmethod
    def get_model(cls, model_type: str) -> BaseTimeSeriesModel:
        """Get a model instance by type"""
        model_class = cls._models.get(model_type.lower())
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")
        return model_class()

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of registered model types"""
        return list(cls._models.keys())

# Register models with factory
ModelFactory.register('xgboost', XGBoostTimeSeriesModel)
ModelFactory.register('decision_tree', DecisionTreeTimeSeriesModel)
# Register models with factory
ModelFactory.register('random_forest', RandomForestTimeSeriesModel)
ModelFactory.register('lstm', LSTMTimeSeriesModel)