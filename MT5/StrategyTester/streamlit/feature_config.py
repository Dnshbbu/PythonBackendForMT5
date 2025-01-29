"""
Centralized feature configuration for ML models.
"""
from typing import List

# Technical Analysis Features
TECHNICAL_FEATURES = [
    'Factors_maScore', 'Factors_rsiScore', 'Factors_macdScore', 'Factors_stochScore',
    'Factors_bbScore', 'Factors_atrScore', 'Factors_sarScore', 'Factors_ichimokuScore',
    'Factors_adxScore', 'Factors_volumeScore', 'Factors_mfiScore', 'Factors_priceMAScore',
    'Factors_emaScore', 'Factors_emaCrossScore', 'Factors_cciScore'
]

# Entry Strategy Features
ENTRY_FEATURES = [
    'EntryScore_AVWAP', 'EntryScore_EMA', 'EntryScore_SR'
]

# Combined features for model training
SELECTED_FEATURES = TECHNICAL_FEATURES + ENTRY_FEATURES

def get_all_features() -> List[str]:
    """Get the complete list of features."""
    return SELECTED_FEATURES

def get_feature_groups() -> dict:
    """Get features organized by groups."""
    return {
        'technical': TECHNICAL_FEATURES,
        'entry': ENTRY_FEATURES
    }