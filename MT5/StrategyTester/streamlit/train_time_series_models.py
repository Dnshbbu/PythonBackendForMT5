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
                      choices=['Auto ARIMA', 'ARIMA', 'SARIMA', 'Prophet', 'VAR'],
                      help='Type of time series model to train')
    parser.add_argument('--model-name', required=True,
                      help='Name for the saved model')
    
    # Optional arguments for data preparation
    parser.add_argument('--n-lags', type=int, default=3,
                      help='Number of lagged features to use (default: 3)')
    parser.add_argument('--prediction-horizon', type=int, default=1,
                      help='Number of steps ahead to predict (default: 1)')
    
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
        
        # Initialize MLflow tracking
        mlflow_manager = MLflowManager()
        run_name = f"{args.model_type}_{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow_manager.start_run(run_name=run_name):
            # Log basic parameters
            mlflow_manager.log_params({
                'model_type': args.model_type,
                'target_column': args.target,
                'feature_columns': args.features,
                'n_lags': args.n_lags,
                'prediction_horizon': args.prediction_horizon,
                'tables': args.tables
            })
            
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
            
            # Train model based on type
            logging.info(f"Training {args.model_type} model...")
            
            if args.model_type == 'Prophet':
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
                
            elif args.model_type == 'VAR':
                # Log VAR-specific parameters
                mlflow_manager.log_params({
                    'maxlags': args.maxlags
                })
                
                # Prepare VAR data
                var_columns = [args.target] + args.features
                var_data = data[var_columns].select_dtypes(include=['float64', 'int64'])
                model, metrics = train_var(var_data, maxlags=args.maxlags)
                
            elif args.model_type == 'Auto ARIMA':
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
                
            elif args.model_type == 'ARIMA':
                if not args.order:
                    raise ValueError("ARIMA order (p,d,q) must be specified")
                
                # Log ARIMA-specific parameters
                mlflow_manager.log_params({
                    'order': args.order
                })
                
                model, metrics = train_arima(future_target, tuple(args.order))
                
            elif args.model_type == 'SARIMA':
                if not args.order or not args.seasonal_order:
                    raise ValueError("Both order and seasonal_order must be specified for SARIMA")
                
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
            model_path = os.path.join(models_dir, args.model_name)
            
            # Prepare metadata based on model type
            metadata = {
                'model_type': args.model_type,
                'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'features': args.features,
                'target': args.target,
                'n_lags': args.n_lags,
                'metrics': metrics,
                'mlflow_run_id': mlflow.active_run().info.run_id if mlflow.active_run() else None
            }
            
            # Add model-specific parameters to metadata
            if args.model_type == 'Auto ARIMA':
                metadata.update({
                    'max_p': args.max_p,
                    'max_d': args.max_d,
                    'max_q': args.max_q,
                    'seasonal': args.seasonal,
                    'seasonal_period': args.seasonal_period
                })
            elif args.model_type in ['ARIMA', 'SARIMA']:
                metadata['order'] = args.order
                if args.model_type == 'SARIMA':
                    metadata['seasonal_order'] = args.seasonal_order
            elif args.model_type == 'Prophet':
                metadata.update({
                    'changepoint_prior_scale': args.changepoint_prior_scale,
                    'seasonality_prior_scale': args.seasonality_prior_scale
                })
            elif args.model_type == 'VAR':
                metadata['maxlags'] = args.maxlags
            
            # Save model locally
            save_model(model, model_path, metadata)
            logging.info(f"Model saved to {model_path}")
            
            # Log model to MLflow
            mlflow_manager.log_model(model, args.model_name)
            
            # Log model metadata as artifact
            metadata_path = f"{args.model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            mlflow_manager.log_artifact(metadata_path)
            os.remove(metadata_path)  # Clean up temporary file
            
            logging.info("Training completed successfully")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 