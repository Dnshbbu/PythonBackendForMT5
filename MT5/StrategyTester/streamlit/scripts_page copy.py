# scripts_page.py
import streamlit as st
import pandas as pd
import os
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

def check_value_ranges(file_path: str):
    """Check value ranges in CSV file."""
    def check_value_range(value):
        try:
            num_value = float(value)
            return -100 <= num_value <= 100
        except (ValueError, TypeError):
            return True

    try:
        df = pd.read_csv(file_path)
        columns_to_check = ['factors', 'score', 'efactors', 'exitScore']
        issues = []
        
        for column in columns_to_check:
            if column not in df.columns:
                continue
                
            for idx, cell in df[column].items():
                if pd.isna(cell) or cell == '':
                    continue
                    
                pairs = cell.split('|')
                
                for pair in pairs:
                    if '=' not in pair:
                        continue
                        
                    name, value = pair.split('=')
                    
                    if not check_value_range(value):
                        issues.append({
                            'row': idx + 2,
                            'column': column,
                            'name': name,
                            'value': value
                        })
        
        return issues
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return []

def preprocess_data(file_path: str) -> None:
    """Preprocess trading data and display results."""
    try:
        # Read and preprocess data
        processed_df = preprocess_trading_data(file_path)
        
        # Display sample of processed data
        st.subheader("Preprocessed Data Preview")
        st.dataframe(processed_df.head())
        
        # Display feature information
        preprocessor = TradingDataPreprocessor()
        preprocessor.preprocess(pd.read_csv(file_path))
        features = preprocessor.get_feature_names()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Numerical Features:", len(features['numerical']))
            st.write(features['numerical'])
        
        with col2:
            st.write("Categorical Features:", len(features['categorical']))
            st.write(features['categorical'])
        
        # Save preprocessed data
        if st.button("Save Preprocessed Data"):
            output_path = file_path.rsplit('.', 1)[0] + '_preprocessed.csv'
            processed_df.to_csv(output_path, index=False)
            st.success(f"Saved preprocessed data to: {output_path}")
            
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")

def scripts():
    """Scripts page implementation"""
    st.title("MT5 Scripts")
    
    file_path = st.text_input("Enter the path to your CSV file:", value="")
    
    if not file_path:
        st.info("Please enter a file path to proceed")
        return
        
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return
    
    st.header("Available Scripts")
    
    # Data Preprocessor
    st.subheader("Trading Data Preprocessor")
    st.write("Preprocess trading data for ML analysis by extracting features from complex columns")
    
    if st.button("Preprocess Trading Data", type="primary"):
        preprocess_data(file_path)
    
    st.markdown("---")
    
    # Value Range Checker
    st.subheader("Value Range Checker")
    st.write("Checks if values in specific columns are within the -100 to +100 range")
    
    if st.button("Run Value Range Check", type="primary"):
        issues = check_value_ranges(file_path)
        
        if issues:
            st.warning("Found values outside the range -100 to +100:")
            
            issues_df = pd.DataFrame(issues)
            st.dataframe(issues_df)
            
            csv = issues_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="value_range_issues.csv",
                mime="text/csv"
            )
        else:
            st.success("No values found outside the range -100 to +100")