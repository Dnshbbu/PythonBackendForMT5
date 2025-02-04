# trading_preprocessor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

class TradingDataPreprocessor:
    """Preprocessor for trading data with complex factor columns."""
    
    def __init__(self):
        self.numerical_features = []
        self.categorical_features = []
    
    @staticmethod
    def _parse_factor_string(factor_str: str) -> Dict[str, float]:
        """Parse string of format 'key1=value1|key2=value2' into a dictionary."""
        if pd.isna(factor_str) or factor_str == '':
            return {}
            
        result = {}
        pairs = factor_str.split('|')
        
        for pair in pairs:
            if '=' not in pair:
                continue
            key, value = pair.split('=')
            try:
                result[key.strip()] = float(value.strip())
            except ValueError:
                result[key.strip()] = value.strip()
                
        return result

    def _extract_features_from_dict(self, data_dict: Dict[str, Union[float, str]], prefix: str = '') -> Dict[str, Union[float, str]]:
        """Extract features from dictionary with proper prefixing."""
        features = {}
        
        for key, value in data_dict.items():
            feature_name = f"{prefix}_{key}" if prefix else key
            
            if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit()):
                features[feature_name] = float(value)
                if feature_name not in self.numerical_features:
                    self.numerical_features.append(feature_name)
            else:
                features[feature_name] = value
                if feature_name not in self.categorical_features:
                    self.categorical_features.append(feature_name)
                
        return features

    def preprocess(self, df: pd.DataFrame, include_categorical: bool = True) -> pd.DataFrame:
        """
        Preprocess the trading data DataFrame.
        
        Args:
            df: Input DataFrame with trading data
            include_categorical: Whether to include categorical features in output
            
        Returns:
            Preprocessed DataFrame with extracted features
        """
        # Reset feature lists
        self.numerical_features = []
        self.categorical_features = []
        
        # Create copy of input DataFrame
        processed_df = df.copy()
        
        # Convert Time to datetime if not already
        processed_df['Time'] = pd.to_datetime(processed_df['Time'])
        
        # Add basic trading features
        processed_df['TradeDirection'] = processed_df['Action'].map({'BUY': 1, 'SELL': -1, 'CLOSE': 0})
        processed_df['TradeDuration'] = processed_df.groupby('Ticket')['Time'].diff().dt.total_seconds()
        
        # Process complex factor columns
        factor_columns = ['factors', 'score', 'efactors', 'exitScore']
        
        for col in factor_columns:
            if col not in processed_df.columns:
                continue
                
            # Parse factor strings into dictionaries
            parsed_dicts = processed_df[col].apply(self._parse_factor_string)
            
            # Extract features from each dictionary
            extracted_features = parsed_dicts.apply(
                lambda x: self._extract_features_from_dict(x, prefix=col)
            )
            
            # Convert to DataFrame and join with main DataFrame
            features_df = pd.DataFrame.from_records(extracted_features.tolist())
            if not features_df.empty:
                processed_df = pd.concat([processed_df.drop(col, axis=1), features_df], axis=1)
        
        # Convert all possible columns to numeric
        for col in processed_df.columns:
            try:
                processed_df[col] = pd.to_numeric(processed_df[col])
            except (ValueError, TypeError):
                continue
        
        # Drop categorical columns if not wanted
        if not include_categorical:
            processed_df = processed_df.drop(columns=self.categorical_features)
        
        return processed_df
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """Get lists of numerical and categorical feature names."""
        return {
            'numerical': self.numerical_features,
            'categorical': self.categorical_features
        }

def preprocess_trading_data(file_path: str, include_categorical: bool = True) -> pd.DataFrame:
    """
    Convenience function to preprocess trading data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        include_categorical: Whether to include categorical features
        
    Returns:
        Preprocessed DataFrame
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Create preprocessor and process data
        preprocessor = TradingDataPreprocessor()
        processed_df = preprocessor.preprocess(df, include_categorical)
        
        return processed_df
        
    except Exception as e:
        raise Exception(f"Error preprocessing trading data: {str(e)}")