import sqlite3
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

class ModelRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path
        logging.info(f"Initializing ModelRepository with db_path: {db_path}")
        self.setup_repository()
        
    def setup_repository(self):
        """Setup the model repository table in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create model repository table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_repository (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT UNIQUE,
                    model_type TEXT,
                    training_type TEXT,
                    prediction_horizon INTEGER,
                    features TEXT,  -- JSON array of feature names
                    target TEXT,  -- Target variable name
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
            
            # Create index on model_name
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_name 
                ON model_repository(model_name)
            """)
            
            conn.commit()
            logging.info("Model repository table setup completed")
            
        except Exception as e:
            logging.error(f"Error setting up model repository: {e}")
            raise
        finally:
            if conn:
                conn.close()
                
    def store_model_info(self, 
                        model_name: str,
                        model_type: str,
                        training_type: str,
                        prediction_horizon: int,
                        features: List[str],
                        target: str,
                        feature_importance: Dict,
                        model_params: Dict,
                        metrics: Dict,
                        training_tables: List[str],
                        training_period: Dict,
                        data_points: int,
                        model_path: str,
                        scaler_path: Optional[str] = None,
                        additional_metadata: Optional[Dict] = None) -> bool:
        """Store model information in the repository"""
        try:
            logging.info(f"Attempting to store model info for: {model_name}")
            logging.info(f"Model type: {model_type}, Training type: {training_type}")
            logging.info(f"Features count: {len(features)}")
            logging.info(f"Training tables: {training_tables}")
            
            # Convert NumPy types to Python native types
            def convert_to_native_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_native_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native_types(item) for item in obj]
                elif isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            # Convert feature importance dictionary
            feature_importance = convert_to_native_types(feature_importance)
            metrics = convert_to_native_types(metrics)
            model_params = convert_to_native_types(model_params)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prepare data for insertion
            data = {
                'model_name': model_name,
                'model_type': model_type,
                'training_type': training_type,
                'prediction_horizon': prediction_horizon,
                'features': json.dumps(features),
                'target': target,
                'feature_importance': json.dumps(feature_importance),
                'model_params': json.dumps(model_params),
                'metrics': json.dumps(metrics),
                'training_tables': json.dumps(training_tables),
                'training_period_start': training_period['start'],
                'training_period_end': training_period['end'],
                'data_points': data_points,
                'model_path': model_path,
                'scaler_path': scaler_path,
                'additional_metadata': json.dumps(additional_metadata) if additional_metadata else None,
                'last_updated': datetime.now().isoformat()
            }
            
            logging.info("Prepared data for insertion:")
            for key, value in data.items():
                logging.debug(f"{key}: {str(value)[:100]}...")  # Show first 100 chars of each value
            
            # Insert or update
            cursor.execute("""
                INSERT OR REPLACE INTO model_repository (
                    model_name, model_type, training_type, prediction_horizon,
                    features, target, feature_importance, model_params,
                    metrics, training_tables, training_period_start, training_period_end,
                    data_points, model_path, scaler_path, additional_metadata,
                    last_updated
                ) VALUES (
                    :model_name, :model_type, :training_type, :prediction_horizon,
                    :features, :target, :feature_importance, :model_params,
                    :metrics, :training_tables, :training_period_start, :training_period_end,
                    :data_points, :model_path, :scaler_path, :additional_metadata,
                    :last_updated
                )
            """, data)
            
            conn.commit()
            logging.info(f"Successfully stored model information for {model_name}")
            
            # Verify the insertion
            cursor.execute("SELECT * FROM model_repository WHERE model_name = ?", (model_name,))
            result = cursor.fetchone()
            if result:
                logging.info("Verified: Model information was stored successfully")
            else:
                logging.warning("Warning: Model information may not have been stored properly")
            
            return True
            
        except Exception as e:
            logging.error(f"Error storing model information: {str(e)}")
            logging.exception("Detailed traceback:")
            return False
        finally:
            if conn:
                conn.close() 

    def get_meta_model_info(self, model_name: str) -> Dict:
        """
        Get meta model information including base model details
        
        Args:
            model_name: Name of the meta model
            
        Returns:
            Dictionary containing meta model information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT model_type, training_type, prediction_horizon,
                       features, feature_importance, model_params,
                       metrics, training_tables, training_period_start,
                       training_period_end, data_points, model_path,
                       scaler_path, additional_metadata
                FROM model_repository
                WHERE model_name = ? AND model_type = 'meta_xgboost'
            """, (model_name,))
            
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Meta model not found: {model_name}")
                
            # Convert row to dictionary
            columns = ['model_type', 'training_type', 'prediction_horizon',
                      'features', 'feature_importance', 'model_params',
                      'metrics', 'training_tables', 'training_period_start',
                      'training_period_end', 'data_points', 'model_path',
                      'scaler_path', 'additional_metadata']
            
            info = dict(zip(columns, result))
            
            # Parse JSON fields
            for field in ['features', 'feature_importance', 'model_params',
                         'metrics', 'training_tables', 'additional_metadata']:
                if info[field]:
                    info[field] = json.loads(info[field])
            
            # Extract base model information
            if info['additional_metadata']:
                info['base_model_run_ids'] = info['additional_metadata'].get('base_model_run_ids', [])
                info['base_model_names'] = info['additional_metadata'].get('base_model_names', [])
            
            return info
            
        except Exception as e:
            logging.error(f"Error retrieving meta model information: {e}")
            raise
        finally:
            if conn:
                conn.close() 

    def get_available_models(self) -> List[Dict]:
        """
        Get list of available models from the repository
        
        Returns:
            List of dictionaries containing model information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT model_name, model_type, training_type, prediction_horizon,
                       features, target, metrics, data_points, model_path,
                       created_at, is_active
                FROM model_repository
                WHERE is_active = 1
                ORDER BY created_at DESC
            """)
            
            results = cursor.fetchall()
            models = []
            
            for row in results:
                model_info = {
                    'name': row[0],
                    'type': row[1],
                    'training_type': row[2],
                    'prediction_horizon': row[3],
                    'features': len(json.loads(row[4])) if row[4] else 1,
                    'target': row[5],
                    'metrics': json.loads(row[6]) if row[6] else {},
                    'data_points': row[7],
                    'model_path': row[8],
                    'created': row[9],
                    'is_active': bool(row[10])
                }
                models.append(model_info)
            
            return models
            
        except Exception as e:
            logging.error(f"Error retrieving available models: {e}")
            return []
        finally:
            if conn:
                conn.close() 

    def get_model_info(self, model_name: str) -> Dict:
        """
        Get model information for any model type
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary containing model information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT model_type, training_type, prediction_horizon,
                       features, target, feature_importance, model_params,
                       metrics, training_tables, training_period_start,
                       training_period_end, data_points, model_path,
                       scaler_path, additional_metadata
                FROM model_repository
                WHERE model_name = ? AND is_active = 1
            """, (model_name,))
            
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Model not found: {model_name}")
                
            # Convert row to dictionary
            columns = ['model_type', 'training_type', 'prediction_horizon',
                      'features', 'target', 'feature_importance', 'model_params',
                      'metrics', 'training_tables', 'training_period_start',
                      'training_period_end', 'data_points', 'model_path',
                      'scaler_path', 'additional_metadata']
            
            info = dict(zip(columns, result))
            
            # Parse JSON fields
            for field in ['features', 'feature_importance', 'model_params',
                         'metrics', 'training_tables', 'additional_metadata']:
                if info[field]:
                    info[field] = json.loads(info[field])
            
            return info
            
        except Exception as e:
            logging.error(f"Error retrieving model information: {e}")
            raise
        finally:
            if conn:
                conn.close() 