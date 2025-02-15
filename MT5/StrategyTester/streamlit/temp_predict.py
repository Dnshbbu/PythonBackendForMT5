import os
import sys
import traceback

try:
    # Change to the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Import required modules
    from model_repository import ModelRepository
    from time_series_predictor import TimeSeriesPredictor
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Get model info
    db_path = os.path.join('logs', 'trading_data.db')
    model_repo = ModelRepository(db_path)
    model_name = 'arima_trip_nas_model_arima'
    
    logging.info(f"Getting model info for: {model_name}")
    model_info = model_repo.get_model_info(model_name)
    logging.info(f"Model info: {model_info}")
    
    # Run prediction command
    cmd = f"python predict_time_series.py --model-name {model_name} --table strategy_TRIP_NAS_10031622 --output-format csv --show-metrics"
    logging.info(f"Running command: {cmd}")
    os.system(cmd)
    
except Exception as e:
    print(f"Error: {str(e)}")
    print("Traceback:")
    traceback.print_exc() 