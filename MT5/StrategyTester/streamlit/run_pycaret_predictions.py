import os
import logging
import argparse
from typing import Optional, Dict, List
from pycaret_model_predictor import PyCaretModelPredictor

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def make_predictions(
    table_name: str,
    model_name: Optional[str] = None,
    n_rows: int = 100
) -> Dict:
    """
    Make predictions using a PyCaret model
    
    Args:
        table_name: Name of the table to get data from
        model_name: Optional name of the model to use (uses latest if not specified)
        n_rows: Number of recent rows to use
        
    Returns:
        Dictionary containing prediction results
    """
    db_path = "trading_data.db"
    models_dir = "models"
    
    predictor = PyCaretModelPredictor(db_path, models_dir)
    
    if model_name:
        predictor.load_model_by_name(model_name)
        logging.info(f"Loaded model: {model_name}")
    else:
        predictor.load_latest_model()
        logging.info(f"Loaded latest model: {predictor.current_model_name}")
    
    # Make predictions
    result = predictor.make_predictions(table_name, n_rows)
    
    # Get explanation
    explanation = predictor.get_prediction_explanation(result)
    
    logging.info("Prediction Results:")
    logging.info(f"Prediction: {result['prediction']:.4f}")
    logging.info(f"Model: {result['model_name']}")
    logging.info(f"Timestamp: {result['timestamp']}")
    logging.info("\nExplanation:")
    logging.info(explanation)
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained PyCaret models')
    
    parser.add_argument(
        '--table',
        help='Table to get data from',
        required=True
    )
    
    parser.add_argument(
        '--model',
        help='Name of the model to use (optional, uses latest if not specified)',
        default=None
    )
    
    parser.add_argument(
        '--rows',
        type=int,
        help='Number of recent rows to use',
        default=100
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        make_predictions(
            table_name=args.table,
            model_name=args.model,
            n_rows=args.rows
        )
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main() 