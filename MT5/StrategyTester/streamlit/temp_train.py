import os
import sys

# Change to the correct directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Construct the command
command = [
    "python",
    "train_time_series_models.py",
    "--tables", "strategy_TRIP_NAS_10031622",
    "--target", "Price",
    "--features",
    "Factors_rsiScore", "Factors_macdScore", "Factors_stochScore",
    "Factors_bbScore", "Factors_atrScore", "Factors_sarScore",
    "Factors_ichimokuScore", "Factors_adxScore", "Factors_volumeScore",
    "Factors_mfiScore", "Factors_priceMAScore", "Factors_emaScore",
    "Factors_cciScore", "EntryScore_AVWAP", "EntryScore_EMA",
    "EntryScore_SR", "EntryScore_Pullback",
    "--model-type", "ARIMA",
    "--model-name", "arima_trip_nas_model",
    "--order", "1", "1", "1"
]

# Execute the command
os.system(" ".join(command)) 