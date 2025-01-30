import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import logging
from typing import Dict, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class PredictionTracker:
    def __init__(self, db_path: str, max_history: int = 1000):
        """
        Initialize PredictionTracker
        
        Args:
            db_path: Path to SQLite database
            max_history: Maximum number of predictions to keep in memory
        """
        self.db_path = db_path
        self.max_history = max_history
        self.setup_logging()
        self.setup_database()
        
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def setup_database(self):
        """Setup database tables for prediction tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    run_id TEXT,
                    actual_price REAL,
                    predicted_price REAL,
                    confidence REAL,
                    model_name TEXT,
                    prediction_error REAL,
                    is_confident BOOLEAN
                )
            """)
            
            # Create metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    run_id TEXT,
                    window_size INTEGER,
                    rmse REAL,
                    mae REAL,
                    r2 REAL,
                    accuracy_rate REAL,
                    confident_accuracy_rate REAL
                )
            """)
            
            conn.commit()
            
        except Exception as e:
            logging.error(f"Error setting up database: {e}")
            raise
        finally:
            if conn:
                conn.close()
                
    def record_prediction(self, 
                         run_id: str,
                         actual_price: float,
                         prediction_data: Dict):
        """
        Record a new prediction and its actual value
        
        Args:
            run_id: Strategy run identifier
            actual_price: Actual price value
            prediction_data: Dictionary containing prediction details
        """
        try:
            prediction = prediction_data['prediction']
            confidence = prediction_data['confidence']
            is_confident = prediction_data['is_confident']
            model_name = prediction_data['model_info']['current_model']
            
            # Calculate prediction error
            prediction_error = actual_price - prediction
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert prediction record
            cursor.execute("""
                INSERT INTO prediction_history (
                    timestamp, run_id, actual_price, predicted_price,
                    confidence, model_name, prediction_error, is_confident
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                run_id,
                actual_price,
                prediction,
                confidence,
                model_name,
                prediction_error,
                is_confident
            ))
            
            # Calculate and store metrics for different windows
            for window_size in [10, 50, 100]:
                self.calculate_and_store_metrics(cursor, run_id, window_size)
            
            conn.commit()
            
        except Exception as e:
            logging.error(f"Error recording prediction: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def calculate_and_store_metrics(self, cursor, run_id: str, window_size: int):
        """Calculate and store metrics for a specific window size"""
        try:
            # Get recent predictions
            cursor.execute("""
                SELECT actual_price, predicted_price, is_confident
                FROM prediction_history
                WHERE run_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (run_id, window_size))
            
            results = cursor.fetchall()
            if len(results) < window_size:
                return
                
            # Convert to arrays
            actuals = np.array([r[0] for r in results])
            predictions = np.array([r[1] for r in results])
            confident_mask = np.array([r[2] for r in results])
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            
            # Calculate accuracy rates
            correct_direction = np.sign(predictions[1:] - predictions[:-1]) == \
                              np.sign(actuals[1:] - actuals[:-1])
            accuracy_rate = np.mean(correct_direction)
            
            # Calculate accuracy rate for confident predictions only
            confident_correct = correct_direction[confident_mask[1:]]
            confident_accuracy_rate = np.mean(confident_correct) if len(confident_correct) > 0 else None
            
            # Store metrics
            cursor.execute("""
                INSERT INTO prediction_metrics (
                    timestamp, run_id, window_size, rmse, mae, r2,
                    accuracy_rate, confident_accuracy_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                run_id,
                window_size,
                rmse,
                mae,
                r2,
                accuracy_rate,
                confident_accuracy_rate
            ))
            
        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")
            raise
            
    def get_latest_metrics(self, run_id: str) -> Dict:
        """Get the latest metrics for all window sizes"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            metrics = {}
            for window_size in [10, 50, 100]:
                cursor.execute("""
                    SELECT rmse, mae, r2, accuracy_rate, confident_accuracy_rate
                    FROM prediction_metrics
                    WHERE run_id = ? AND window_size = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (run_id, window_size))
                
                result = cursor.fetchone()
                if result:
                    metrics[f'window_{window_size}'] = {
                        'rmse': result[0],
                        'mae': result[1],
                        'r2': result[2],
                        'accuracy_rate': result[3],
                        'confident_accuracy_rate': result[4]
                    }
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error getting metrics: {e}")
            raise
        finally:
            if conn:
                conn.close()