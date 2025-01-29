import os
import logging
import sqlite3
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any
import joblib
import json
import numpy as np
from model_manager import ModelManager
from model_trainer import TimeSeriesModelTrainer
from feature_config import SELECTED_FEATURES

class ModelTrainingManager:
    def verify_database_schema(self):
        """Verify and repair database schema if needed"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check model_training_status table
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='model_training_status'
            """)
            exists = cursor.fetchone() is not None
            
            if exists:
                # Drop and recreate if schema is incorrect
                cursor.execute("PRAGMA table_info(model_training_status)")
                columns = {col[1] for col in cursor.fetchall()}
                required_columns = {
                    'id', 'start_time', 'end_time', 'status', 'model_path',
                    'metrics', 'error_message', 'rows_processed', 'last_processed_id'
                }
                
                if not required_columns.issubset(columns):
                    logging.warning("Incorrect schema found, recreating model_training_status table")
                    cursor.execute("DROP TABLE model_training_status")
                    conn.commit()
                    
                    # Recreate table
                    self.setup_training_table()
            else:
                self.setup_training_table()
            
            logging.info("Database schema verification completed")
            
        except Exception as e:
            logging.error(f"Error verifying database schema: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def __init__(self, db_path: str, models_dir: str, min_rows_for_training: int = 20):
        """
        Initialize the ModelTrainingManager
        
        Args:
            db_path: Path to SQLite database
            models_dir: Directory to save trained models
            min_rows_for_training: Minimum number of new rows before triggering retraining
        """
        self.db_path = db_path
        self.models_dir = models_dir
        self.min_rows_for_training = min_rows_for_training
        self.training_lock = threading.Lock()
        self.is_training = False
        self.setup_logging()
        self.verify_database_schema()  # Verify schema on initialization
        self.model_manager = ModelManager()
        
    def _serialize_metrics(self, metrics: Dict) -> str:
        """Safely serialize metrics to JSON string"""
        if not metrics:
            return "{}"
            
        # Convert any non-serializable objects to strings
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                serializable_metrics[key] = value
            else:
                serializable_metrics[key] = str(value)
                
        try:
            return json.dumps(serializable_metrics)
        except Exception as e:
            logging.error(f"Error serializing metrics: {e}")
            return "{}"

    def cleanup_invalid_metrics(self):
        """Clean up any invalid JSON in the metrics column"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all rows
            cursor.execute("SELECT id, metrics FROM model_training_status")
            rows = cursor.fetchall()
            
            for row_id, metrics in rows:
                if metrics:
                    try:
                        # Try to parse the JSON
                        if isinstance(metrics, str):
                            json.loads(metrics.strip())
                    except json.JSONDecodeError:
                        # If invalid JSON, update with empty object
                        cursor.execute(
                            "UPDATE model_training_status SET metrics = ? WHERE id = ?",
                            ("{}", row_id)
                        )
                        logging.info(f"Cleaned invalid metrics for row {row_id}")
            
            conn.commit()
            
        except Exception as e:
            logging.error(f"Error cleaning up metrics: {e}")
        finally:
            if conn:
                conn.close()

    def get_feature_config(self) -> dict:
        """Get the current feature configuration"""
        try:
            from xgboost_train_model import (
                TECHNICAL_FEATURES,
                ENTRY_FEATURES
            )
            
            return {
                'technical_features': TECHNICAL_FEATURES,
                'entry_features': ENTRY_FEATURES,
                'all_features': TECHNICAL_FEATURES + ENTRY_FEATURES
            }
        except ImportError as e:
            logging.error(f"Error importing feature configuration: {e}")
            return {
                'technical_features': [],
                'entry_features': [],
                'all_features': []
            }

    def setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def setup_training_table(self):
        """Create table to track model training status"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table to track model training status
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_training_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT CHECK(status IN ('started', 'completed', 'failed')),
                    model_path TEXT,
                    metrics TEXT CHECK(metrics IS NULL OR json_valid(metrics)),
                    error_message TEXT,
                    rows_processed INTEGER,
                    last_processed_id INTEGER
                )
            """)
            
            # Create table to track data processing status
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_processing_status (
                    table_name TEXT PRIMARY KEY,
                    last_processed_id INTEGER,
                    total_rows INTEGER,
                    last_update TIMESTAMP
                )
            """)
            
            conn.commit()
            
        except Exception as e:
            logging.error(f"Error setting up training table: {e}")
            raise
        finally:
            if conn:
                conn.close()


    # def get_unprocessed_row_count(self, table_name: str) -> int:
    #     """Get count of unprocessed rows for a table"""
    #     try:
    #         conn = sqlite3.connect(self.db_path)
    #         cursor = conn.cursor()
            
    #         # Get last processed ID
    #         cursor.execute("""
    #             SELECT last_processed_id 
    #             FROM data_processing_status 
    #             WHERE table_name = ?
    #         """, (table_name,))
    #         result = cursor.fetchone()
            
    #         last_id = result[0] if result else 0
            
    #         # Get current max ID from data table
    #         cursor.execute(f"SELECT MAX(id) FROM {table_name}")
    #         max_id = cursor.fetchone()[0] or 0
            
    #         return max_id - last_id
            
    #     except Exception as e:
    #         logging.error(f"Error getting unprocessed row count: {e}")
    #         raise
    #     finally:
    #         if conn:
    #             conn.close()

    def get_unprocessed_row_count(self, table_name: str) -> int:
        """Get count of unprocessed rows for a table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Sanitize table name
            sanitized_table = table_name.replace('.', '_').replace(' ', '_').replace('-', '_')
            
            cursor.execute("""
                SELECT last_processed_id 
                FROM data_processing_status 
                WHERE table_name = ?
            """, (sanitized_table,))
            result = cursor.fetchone()
            
            last_id = result[0] if result else 0
            
            cursor.execute(f"SELECT MAX(id) FROM '{sanitized_table}'")
            max_id = cursor.fetchone()[0] or 0
            
            return max_id - last_id
            
        except Exception as e:
            logging.error(f"Error getting unprocessed row count: {e}")
            raise
        finally:
            if conn:
                conn.close()



    # def update_processing_status(self, table_name: str, last_processed_id: int):
    #     """Update the processing status for a table"""
    #     try:
    #         conn = sqlite3.connect(self.db_path)
    #         cursor = conn.cursor()
            
    #         cursor.execute("""
    #             INSERT OR REPLACE INTO data_processing_status 
    #             (table_name, last_processed_id, total_rows, last_update)
    #             VALUES (?, ?, ?, ?)
    #         """, (table_name, last_processed_id, last_processed_id, datetime.now()))
            
    #         conn.commit()
            
    #     except Exception as e:
    #         logging.error(f"Error updating processing status: {e}")
    #         raise
    #     finally:
    #         if conn:
    #             conn.close()


    def update_processing_status(self, table_name: str, last_processed_id: int):
        """Update the processing status for a table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Sanitize table name
            sanitized_table = table_name.replace('.', '_').replace(' ', '_').replace('-', '_')
            
            cursor.execute("""
                INSERT OR REPLACE INTO data_processing_status 
                (table_name, last_processed_id, total_rows, last_update)
                VALUES (?, ?, ?, ?)
            """, (sanitized_table, last_processed_id, last_processed_id, datetime.now()))
            
            conn.commit()
            
        except Exception as e:
            logging.error(f"Error updating processing status: {e}")
            raise
        finally:
            if conn:
                conn.close()


    def start_training_session(self) -> int:
        """Start a new training session and return its ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO model_training_status 
                (start_time, status)
                VALUES (?, ?)
            """, (datetime.now(), 'started'))
            
            session_id = cursor.lastrowid
            conn.commit()
            return session_id
            
        except Exception as e:
            logging.error(f"Error starting training session: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def _validate_and_serialize_metrics(self, metrics: Optional[Dict]) -> str:
        """Validate and serialize metrics to a JSON string"""
        if not metrics:
            return "{}"
            
        def convert_value(v):
            if isinstance(v, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                            np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(v)
            elif isinstance(v, (np.float_, np.float16, np.float32, np.float64)):
                return float(v)
            elif isinstance(v, (np.ndarray, list)):
                return [convert_value(x) for x in v]
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            return v

    def update_training_status(self, session_id: int, status: str, 
                             model_path: Optional[str] = None,
                             metrics: Optional[Dict] = None,
                             error_message: Optional[str] = None,
                             rows_processed: Optional[int] = None,
                             last_processed_id: Optional[int] = None):
        """Update the status of a training session"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            update_fields = ['status']
            update_values = [status]
            
            if model_path:
                update_fields.append('model_path')
                update_values.append(model_path)
            
            if metrics is not None:
                update_fields.append('metrics')
                metrics_json = self._validate_and_serialize_metrics(metrics)
                update_values.append(metrics_json)
            
            if error_message:
                update_fields.append('error_message')
                update_values.append(error_message)
            
            if rows_processed is not None:
                update_fields.append('rows_processed')
                update_values.append(rows_processed)
                
            if last_processed_id is not None:
                update_fields.append('last_processed_id')
                update_values.append(last_processed_id)
            
            if status in ['completed', 'failed']:
                update_fields.append('end_time')
                update_values.append(datetime.now().isoformat())
            
            query = f"""
                UPDATE model_training_status 
                SET {', '.join(f'{field} = ?' for field in update_fields)}
                WHERE id = ?
            """
            update_values.append(session_id)
            
            cursor.execute(query, update_values)
            conn.commit()
            
        except Exception as e:
            logging.error(f"Error updating training status: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def check_and_trigger_training(self, table_name: str) -> bool:
        """Check if retraining is needed and trigger if necessary"""
        try:
            # Check if already training
            if self.is_training:
                return False
            
            # Get unprocessed row count
            unprocessed_rows = self.get_unprocessed_row_count(table_name)
            
            if unprocessed_rows >= self.min_rows_for_training:
                # Start training in a separate thread
                training_thread = threading.Thread(
                    target=self.retrain_model,
                    args=(table_name,)
                )
                training_thread.start()
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Error checking training trigger: {e}")
            return False

    def retrain_model(self, table_name: str):
        """Retrain the model with new data"""
        if not self.training_lock.acquire(blocking=False):
            logging.info("Another training session is in progress")
            return
        
        conn = None
        try:
            self.is_training = True
            session_id = self.start_training_session()
            
            # Initialize model manager if available
            try:
                from model_manager import ModelManager
                model_manager = ModelManager()
                model_type = model_manager.get_active_model()
                model_params = model_manager.get_model_params()
            except ImportError:
                model_type = "xgboost"
                from xgboost_train_model import get_model_params
                model_params = get_model_params(model_type)
            
            # # Import feature definitions
            # from xgboost_train_model import (
            #     TECHNICAL_FEATURES,
            #     ENTRY_FEATURES
            # )
            
            # # Combine all features
            # selected_features = TECHNICAL_FEATURES + ENTRY_FEATURES
            selected_features = SELECTED_FEATURES
            
            # Initialize trainer with support for multiple model types
            try:
                from model_trainer import TimeSeriesModelTrainer
                trainer = TimeSeriesModelTrainer(self.db_path, self.models_dir)
            except ImportError:
                from xgboost_trainer import TimeSeriesXGBoostTrainer
                trainer = TimeSeriesXGBoostTrainer(self.db_path, self.models_dir)
            
            # Get last processed ID
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT last_processed_id 
                FROM data_processing_status 
                WHERE table_name = ?
            """, (table_name,))
            result = cursor.fetchone()
            last_processed_id = result[0] if result else 0
            
            # Train model
            try:
                if isinstance(trainer, TimeSeriesModelTrainer):
                    model_path, metrics = trainer.train_and_save(
                        table_name=table_name,
                        model_type=model_type,
                        target_col="Price",
                        feature_cols=selected_features,
                        prediction_horizon=1,
                        model_params=model_params
                    )
                else:
                    # Legacy training for XGBoost trainer
                    model_path, metrics = trainer.train_and_save(
                        table_name=table_name,
                        target_col="Price",
                        prediction_horizon=1,
                        feature_cols=selected_features,
                        model_params=model_params
                    )
            except Exception as train_error:
                logging.error(f"Training error: {train_error}")
                self.update_training_status(
                    session_id=session_id,
                    status='failed',
                    error_message=str(train_error)
                )
                raise
            
            # Get new last processed ID
            cursor.execute(f"SELECT MAX(id) FROM {table_name}")
            new_last_id = cursor.fetchone()[0]
            
            # Update status
            self.update_training_status(
                session_id=session_id,
                status='completed',
                model_path=model_path,
                metrics=metrics,
                rows_processed=new_last_id - last_processed_id if new_last_id else 0,
                last_processed_id=new_last_id
            )
            
            # Update processing status
            self.update_processing_status(table_name, new_last_id)
            
            logging.info(f"Model retraining completed successfully: {model_path}")
            logging.info(f"Used features: {selected_features}")
            
        except Exception as e:
            logging.error(f"Error in model retraining: {e}")
            self.update_training_status(
                session_id=session_id,
                status='failed',
                error_message=str(e)
            )
        finally:
            self.is_training = False
            self.training_lock.release()
            if conn:
                conn.close()

    def get_latest_training_status(self) -> Dict[str, Any]:
        """Get the status of the latest training session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, start_time, end_time, status, model_path, 
                       metrics, error_message, rows_processed
                FROM model_training_status
                ORDER BY id DESC
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            
            if result:
                return {
                    'session_id': result[0],
                    'start_time': result[1],
                    'end_time': result[2],
                    'status': result[3],
                    'model_path': result[4],
                    'metrics': json.loads(result[5]) if result[5] and result[5] != '{}' else None,
                    'error_message': result[6],
                    'rows_processed': result[7]
                }
            return {}
            
        except Exception as e:
            logging.error(f"Error getting training status: {e}")
            raise
        finally:
            if conn:
                conn.close()