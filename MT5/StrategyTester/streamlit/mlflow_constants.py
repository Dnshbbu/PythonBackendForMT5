"""Constants for MLflow logging standardization"""

# Common parameters for both training and prediction
COMMON_PARAMS = {
    'model_type': 'Type of the model (e.g., SARIMA, ARIMA, Prophet, VAR)',
    'target_col': 'Target column for prediction',
    'n_lags': 'Number of lagged features used',
    'data_points': 'Number of data points used',
}

# Training-specific parameters
TRAINING_PARAMS = {
    **COMMON_PARAMS,
    'table_names': 'Comma-separated list of training tables',
    'feature_columns': 'List of feature columns used for training',
    'prediction_horizon': 'Number of steps ahead to predict',
    'training_type': 'Type of training (single, multi, base, incremental)',
    'training_period_start': 'Start date of training data',
    'training_period_end': 'End date of training data',
}

# Model-specific parameters
MODEL_SPECIFIC_PARAMS = {
    'SARIMA': {
        'order': 'ARIMA order (p,d,q)',
        'seasonal_order': 'Seasonal order (P,D,Q,s)',
    },
    'ARIMA': {
        'order': 'ARIMA order (p,d,q)',
    },
    'Prophet': {
        'changepoint_prior_scale': 'Changepoint prior scale',
        'seasonality_prior_scale': 'Seasonality prior scale',
    },
    'VAR': {
        'maxlags': 'Maximum number of lags',
    },
}

# Prediction-specific parameters
PREDICTION_PARAMS = {
    **COMMON_PARAMS,
    'model_name': 'Name of the model used for prediction',
    'source_table': 'Source table for prediction data',
    'run_id': 'Unique identifier for prediction run',
    'prediction_period_start': 'Start date of prediction period',
    'prediction_period_end': 'End date of prediction period',
}

# Common metrics for both training and prediction
COMMON_METRICS = {
    'mean_absolute_error': 'Mean Absolute Error',
    'root_mean_squared_error': 'Root Mean Squared Error',
    'mean_absolute_percentage_error': 'Mean Absolute Percentage Error',
    'r_squared': 'R-squared score',
}

# Training-specific metrics
TRAINING_METRICS = {
    **COMMON_METRICS,
    'training_time': 'Total training time in seconds',
    'convergence_status': 'Model convergence status',
}

# Prediction-specific metrics
PREDICTION_METRICS = {
    **COMMON_METRICS,
    'direction_accuracy': 'Accuracy of predicted price direction',
    'up_prediction_accuracy': 'Accuracy of upward price predictions',
    'down_prediction_accuracy': 'Accuracy of downward price predictions',
    'total_predictions': 'Total number of predictions made',
    'max_error': 'Maximum prediction error',
    'min_error': 'Minimum prediction error',
    'std_error': 'Standard deviation of prediction error',
    'avg_price_change': 'Average price change',
    'price_volatility': 'Price volatility',
    'mean_prediction_error': 'Mean prediction error',
    'median_prediction_error': 'Median prediction error',
    'error_skewness': 'Skewness of prediction errors',
    'first_quarter_accuracy': 'Prediction accuracy in first quarter',
    'last_quarter_accuracy': 'Prediction accuracy in last quarter',
    'max_correct_streak': 'Maximum streak of correct predictions',
    'avg_correct_streak': 'Average streak of correct predictions',
}

# MLflow experiment names
EXPERIMENT_NAMES = {
    'training': 'trading_models',
    'prediction': 'model_predictions',
}

# Run name formats
RUN_NAME_FORMATS = {
    'training': '{model_type}_{training_type}_{timestamp}',  # e.g., ts-sarima_single_20250215_131902
    'prediction': 'run_{timestamp}',  # e.g., run_20250215_131502_940
} 