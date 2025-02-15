import argparse
import logging
import os
import pandas as pd
from datetime import datetime
from time_series_trainer import (
    auto_arima, train_arima, train_sarima, train_prophet, train_var,
    prepare_time_series_data, combine_tables_data, save_model
)
from mlflow_utils import MLflowManager
import json
import mlflow
from model_repository import ModelRepository
import numpy as np

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train time series models from command line')
    
    # Required arguments
    parser.add_argument('--tables', nargs='+', required=True,
                      help='List of table names to use for training')
    parser.add_argument('--target', required=True,
                      help='Target column for prediction')
    parser.add_argument('--features', nargs='+', required=True,
                      help='List of feature columns to use')
    parser.add_argument('--model-type', required=True, 
                      choices=['Auto ARIMA', 'ARIMA', 'SARIMA', 'Prophet', 'VAR', 'multiple'],
                      help='Type of time series model to train')
    parser.add_argument('--model-name', required=False,
                      help='Optional name for the saved model. If not provided, will be auto-generated.')
    
    # Optional arguments for data preparation
    parser.add_argument('--n-lags', type=int, default=3,
                      help='Number of lagged features to use (default: 3)')
    parser.add_argument('--prediction-horizon', type=int, default=1,
                      help='Number of steps ahead to predict (default: 1)')
    
    # Multiple model selection
    parser.add_argument('--selected-models', nargs='+', required=False,
                      choices=['Auto ARIMA', 'ARIMA', 'SARIMA', 'Prophet', 'VAR'],
                      help='List of models to train when model-type is "multiple"')
    
    # Model-specific arguments
    # Auto ARIMA
    parser.add_argument('--max-p', type=int, default=5,
                      help='Maximum P (AR order) for Auto ARIMA (default: 5)')
    parser.add_argument('--max-d', type=int, default=2,
                      help='Maximum D (difference order) for Auto ARIMA (default: 2)')
    parser.add_argument('--max-q', type=int, default=5,
                      help='Maximum Q (MA order) for Auto ARIMA (default: 5)')
    parser.add_argument('--seasonal', action='store_true',
                      help='Include seasonal components in Auto ARIMA')
    parser.add_argument('--seasonal-period', type=int, default=5,
                      help='Seasonal period for Auto ARIMA (default: 5)')
    
    # ARIMA
    parser.add_argument('--order', nargs=3, type=int,
                      help='ARIMA order (p,d,q) as three integers')
    
    # SARIMA
    parser.add_argument('--seasonal-order', nargs=4, type=int,
                      help='SARIMA seasonal order (P,D,Q,s) as four integers')
    
    # Prophet
    parser.add_argument('--changepoint-prior-scale', type=float, default=0.05,
                      help='Changepoint prior scale for Prophet (default: 0.05)')
    parser.add_argument('--seasonality-prior-scale', type=float, default=10.0,
                      help='Seasonality prior scale for Prophet (default: 10.0)')
    
    # VAR
    parser.add_argument('--maxlags', type=int, default=5,
                      help='Maximum lags for VAR model (default: 5)')
    
    return parser.parse_args()

def main():
    """Main function to train time series models"""
    args = parse_args()
    setup_logging()
    
    try:
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'logs', 'trading_data.db')
        models_dir = os.path.join(current_dir, 'models', 'time_series')
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize MLflow tracking with absolute path
        mlflow_db = "C:\\Users\\StdUser\\Desktop\\MyProjects\\Backtesting\\MT5\\StrategyTester\\streamlit\\mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
        
        # Initialize MLflow manager
        mlflow_manager = MLflowManager()
        
        # Load and combine data
        logging.info(f"Loading data from {len(args.tables)} tables")
        df = combine_tables_data(db_path, args.tables)
        
        # Set DateTime as index
        df = df.set_index('DateTime')
        
        # Prepare data for training
        data, future_target, features = prepare_time_series_data(
            df,
            args.target,
            args.features,
            prediction_horizon=args.prediction_horizon,
            n_lags=args.n_lags
        )
        
        logging.info(f"Using features: {features.columns.tolist() if features is not None else []}")
        logging.info(f"Number of lagged features: {args.n_lags}")
        
        # Determine which models to train
        models_to_train = []
        if args.model_type == 'multiple':
            if args.selected_models:
                # Use specified models
                models_to_train = args.selected_models
                logging.info(f"Training selected models: {', '.join(models_to_train)}")
            else:
                # Train all available models if none specified
                models_to_train = ['Auto ARIMA', 'ARIMA', 'SARIMA', 'Prophet', 'VAR']
                logging.info("No models specified, training all available models")
        else:
            models_to_train = [args.model_type]
        
        # Train each model
        for model_type in models_to_train:
            try:
                # Generate model name for each model
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                # Model name format: {model_type}_{training_type}_{timestamp}
                # Convert model type to lowercase and remove spaces
                model_type_clean = model_type.lower().replace(' ', '')
                model_name = f"ts-{model_type_clean}_single_{timestamp}"
                
                # Run ID format: run_YYYYMMDD_HHMMSS_fff
                current_time = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]  # Include milliseconds but truncate to 3 digits
                run_id = f"run_{current_time}"
                
                with mlflow_manager.start_run(run_name=model_name):
                    logging.info(f"\nTraining {model_type} model...")
                    
                    # Log basic parameters
                    mlflow_manager.log_params({
                        'model_type': model_type,
                        'target_col': args.target,
                        'feature_columns': args.features,
                        'n_lags': args.n_lags,
                        'prediction_horizon': args.prediction_horizon,
                        'table_names': ','.join(args.tables)
                    })
                    
                    # Train model based on type
                    if model_type == 'Prophet':
                        # Log Prophet-specific parameters
                        mlflow_manager.log_params({
                            'changepoint_prior_scale': args.changepoint_prior_scale,
                            'seasonality_prior_scale': args.seasonality_prior_scale
                        })
                        
                        # Prepare Prophet data
                        prophet_df = pd.DataFrame({
                            'ds': data.index,
                            'y': future_target
                        })
                        if features is not None:
                            for feature in args.features:
                                if feature != args.target:
                                    prophet_df[feature] = features[feature]
                        
                        model, metrics = train_prophet(
                            prophet_df,
                            args.features
                        )
                        
                    elif model_type == 'VAR':
                        # Log VAR-specific parameters
                        mlflow_manager.log_params({
                            'maxlags': args.maxlags
                        })
                        
                        # Prepare VAR data
                        var_columns = [args.target] + args.features
                        var_data = data[var_columns].select_dtypes(include=['float64', 'int64'])
                        model, metrics = train_var(var_data, maxlags=args.maxlags)
                        
                    elif model_type == 'Auto ARIMA':
                        # Log Auto ARIMA-specific parameters
                        mlflow_manager.log_params({
                            'max_p': args.max_p,
                            'max_d': args.max_d,
                            'max_q': args.max_q,
                            'seasonal': args.seasonal,
                            'seasonal_period': args.seasonal_period
                        })
                        
                        model, metrics = auto_arima(
                            future_target,
                            max_p=args.max_p,
                            max_d=args.max_d,
                            max_q=args.max_q,
                            seasonal=args.seasonal,
                            m=args.seasonal_period if args.seasonal else 1
                        )
                        
                    elif model_type == 'ARIMA':
                        if not args.order:
                            args.order = (1, 1, 1)  # Default order for multiple model training
                        
                        # Log ARIMA-specific parameters
                        mlflow_manager.log_params({
                            'order': args.order
                        })
                        
                        model, metrics = train_arima(future_target, tuple(args.order))
                        
                    elif model_type == 'SARIMA':
                        if not args.order or not args.seasonal_order:
                            args.order = (1, 1, 1)  # Default order
                            args.seasonal_order = (1, 1, 1, 12)  # Default seasonal order
                        
                        # Log SARIMA-specific parameters
                        mlflow_manager.log_params({
                            'order': args.order,
                            'seasonal_order': args.seasonal_order
                        })
                        
                        model, metrics = train_sarima(
                            future_target,
                            tuple(args.order),
                            tuple(args.seasonal_order)
                        )
                    
                    # Log metrics
                    mlflow_manager.log_metrics(metrics)
                    
                    # Save model and metadata
                    model_path = os.path.join(models_dir, model_name)
                    
                    # Prepare metadata based on model type
                    selected_features = args.features if features is None else features.columns.tolist()
                    order = args.order if model_type in ['ARIMA', 'SARIMA'] else None
                    seasonal_order = args.seasonal_order if model_type == 'SARIMA' else None
                    
                    # Prepare training period information
                    training_period = {
                        'start': data.index[0].strftime('%Y-%m-%d %H:%M:%S') if data is not None and len(data) > 0 else None,
                        'end': data.index[-1].strftime('%Y-%m-%d %H:%M:%S') if data is not None and len(data) > 0 else None
                    }
                    
                    # Prepare feature importance
                    feature_importance = {}
                    if model_type == 'VAR' and hasattr(model, 'coefs'):
                        # For VAR models, calculate feature importance based on coefficient magnitudes
                        feature_importance = {
                            feature: float(np.abs(model.coefs[0]).mean()) 
                            for feature in selected_features
                        }
                    else:
                        feature_importance = {
                            'ar_coefficients': [],
                            'ma_coefficients': [],
                            'seasonal_ar_coefficients': [],
                            'seasonal_ma_coefficients': []
                        }
                    
                    # Model parameters
                    model_params = {
                        'order': order,
                        'seasonal_order': seasonal_order,
                        'seasonal': True if seasonal_order else False,
                        'seasonal_period': seasonal_order[3] if seasonal_order else None
                    }
                    
                    # Add model-specific parameters
                    if model_type == 'Auto ARIMA':
                        model_params.update({
                            'max_p': args.max_p,
                            'max_d': args.max_d,
                            'max_q': args.max_q,
                            'seasonal': args.seasonal,
                            'seasonal_period': args.seasonal_period
                        })
                    elif model_type == 'Prophet':
                        model_params.update({
                            'changepoint_prior_scale': args.changepoint_prior_scale,
                            'seasonality_prior_scale': args.seasonality_prior_scale
                        })
                    elif model_type == 'VAR':
                        model_params.update({
                            'maxlags': args.maxlags,
                            'ic': float(model.k_ar) if hasattr(model, 'k_ar') else None,
                            'trend': model.trend if hasattr(model, 'trend') else None
                        })
                    
                    # Additional metadata
                    additional_metadata = {
                        'stationarity_test': {
                            'adf_statistic': None,
                            'adf_pvalue': None
                        },
                        'model_specification': {
                            'trend_specification': None,
                            'has_seasonal': True if seasonal_order else False,
                            'seasonal_period': seasonal_order[3] if seasonal_order else None
                        },
                        'estimation_details': {
                            'nobs': len(data) if data is not None else 0,
                            'df_model': len(selected_features),
                            'df_resid': None
                        }
                    }
                    
                    if model_type == 'VAR':
                        additional_metadata.update({
                            'preprocessing': {
                                'diff_orders': model.diff_orders if hasattr(model, 'diff_orders') else None,
                                'has_scaler': hasattr(model, 'scaler')
                            }
                        })
                    
                    # Prepare complete metadata
                    metadata = {
                        'model_type': model_type,
                        'training_type': 'single',
                        'prediction_horizon': args.prediction_horizon,
                        'features': selected_features,
                        'target': args.target,
                        'n_lags': args.n_lags,
                        'order': order,
                        'seasonal_order': seasonal_order,
                        'seasonal': True if seasonal_order else False,
                        'seasonal_period': seasonal_order[3] if seasonal_order else None,
                        'table_names': args.tables,
                        'data_points': len(data) if data is not None else 0,
                        'training_period': training_period,
                        'model_params': model_params,
                        'metrics': metrics,
                        'feature_importance': feature_importance,
                        'additional_metadata': additional_metadata
                    }
                    
                    # Save model and get model path
                    model_file_path, scaler_path = save_model(model, model_path, metadata)
                    
                    # Initialize model repository
                    model_repo = ModelRepository(db_path)
                    
                    # Store model information
                    model_repo.store_model_info(
                        model_name=model_name,
                        model_type=model_type,
                        training_type='single',
                        prediction_horizon=args.prediction_horizon,
                        features=selected_features,
                        target=args.target,
                        feature_importance=feature_importance,
                        model_params=model_params,
                        metrics=metrics,
                        training_tables=args.tables,
                        training_period=training_period,
                        data_points=len(data) if data is not None else 0,
                        model_path=model_path,
                        scaler_path=scaler_path,
                        additional_metadata=additional_metadata
                    )
                    
                    logging.info(f"Model information stored in repository for {model_name}")
                    logging.info(f"Model saved to {model_file_path}")
                    if scaler_path:
                        logging.info(f"Scaler saved to {scaler_path}")
            
            except Exception as e:
                logging.error(f"Error training {model_type} model: {str(e)}")
                logging.exception("Detailed traceback:")
                continue  # Continue with next model even if one fails
        
        logging.info("Training completed successfully")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        logging.exception("Detailed traceback:")
        raise

if __name__ == "__main__":
    main() 