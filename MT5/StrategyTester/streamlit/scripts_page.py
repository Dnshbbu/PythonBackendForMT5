# scripts_page.py
import streamlit as st
import pandas as pd
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

@dataclass
class PreprocessorConfig:
    """Configuration class for TradingDataPreprocessor"""
    split_columns: List[str]  # Columns to split by delimiter
    delimiter: str = "|"  # Delimiter for splitting
    key_value_separator: str = "="  # Separator for key-value pairs
    datetime_columns: List[str] = None  # Columns to convert to datetime
    prefix_template: str = "{column}_"  # Template for feature name prefix
    output_dir: str = "processed_data"
    save_config: bool = True

class ConfigurableTradingPreprocessor:
    """Configurable preprocessor for trading data with dynamic column splitting."""
    
    def __init__(self, config: PreprocessorConfig):
        self.config = config
        self.numerical_features = []
        self.categorical_features = []
        self._setup_output_directory()
    
    def _setup_output_directory(self):
        """Create output directory if it doesn't exist"""
        # Create absolute path for output directory
        self.config.output_dir = os.path.abspath(self.config.output_dir)
        os.makedirs(self.config.output_dir, exist_ok=True)
        
    def save_configuration(self, filename: str = "preprocessor_config.json"):
        """Save current configuration to JSON"""
        if not self.config.save_config:
            return
            
        config_path = os.path.join(self.config.output_dir, filename)
        config_dict = {
            "split_columns": self.config.split_columns,
            "delimiter": self.config.delimiter,
            "key_value_separator": self.config.key_value_separator,
            "datetime_columns": self.config.datetime_columns,
            "prefix_template": self.config.prefix_template
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @staticmethod
    def load_configuration(config_path: str) -> PreprocessorConfig:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return PreprocessorConfig(**config_dict)

    def _parse_key_value_string(self, value_str: str) -> Dict[str, Any]:
        """Parse string with delimiter into key-value pairs"""
        if pd.isna(value_str) or value_str == '':
            return {}
            
        result = {}
        pairs = value_str.split(self.config.delimiter)
        
        for pair in pairs:
            if self.config.key_value_separator not in pair:
                continue
            key, value = pair.split(self.config.key_value_separator)
            try:
                result[key.strip()] = float(value.strip())
            except ValueError:
                result[key.strip()] = value.strip()
                
        return result

    def _extract_features(self, data_dict: Dict[str, Union[float, str]], prefix: str = '') -> Dict[str, Union[float, str]]:
        """Extract features from dictionary with configurable prefixing"""
        features = {}
        prefix = self.config.prefix_template.format(column=prefix) if prefix else ''
        
        for key, value in data_dict.items():
            feature_name = f"{prefix}{key}"
            
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
        """Preprocess the trading data DataFrame with configurable columns"""
        processed_df = df.copy()
        
        # Reset feature lists
        self.numerical_features = []
        self.categorical_features = []
        
        # Convert datetime columns if specified
        if self.config.datetime_columns:
            for col in self.config.datetime_columns:
                if col in processed_df.columns:
                    processed_df[col] = pd.to_datetime(processed_df[col])
        
        # Process split columns
        for col in self.config.split_columns:
            if col not in processed_df.columns:
                continue
                
            # Parse strings into dictionaries
            parsed_dicts = processed_df[col].apply(self._parse_key_value_string)
            
            # Extract features
            extracted_features = parsed_dicts.apply(
                lambda x: self._extract_features(x, prefix=col)
            )
            
            # Convert to DataFrame and join
            features_df = pd.DataFrame.from_records(extracted_features.tolist())
            if not features_df.empty:
                processed_df = pd.concat([processed_df.drop(col, axis=1), features_df], axis=1)
        
        # Convert numeric columns
        for col in processed_df.columns:
            try:
                processed_df[col] = pd.to_numeric(processed_df[col])
            except (ValueError, TypeError):
                continue
        
        # Handle categorical features
        if not include_categorical:
            processed_df = processed_df.drop(columns=self.categorical_features)
        
        return processed_df
    
    def save_processed_data(self, df: pd.DataFrame, original_filename: str) -> str:
        """Save processed DataFrame with timestamp"""
        try:
            # Ensure output directory exists
            self._setup_output_directory()
            
            # Create timestamp and filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(original_filename))[0]
            output_filename = f"{base_name}_processed_{timestamp}.csv"
            output_path = os.path.join(self.config.output_dir, output_filename)
            
            # Save the DataFrame
            df.to_csv(output_path, index=False)
            
            # Verify the file was created
            if not os.path.exists(output_path):
                raise Exception(f"Failed to create output file: {output_path}")
                
            return output_path
        except Exception as e:
            raise Exception(f"Error saving processed data: {str(e)}")
    
    def get_feature_info(self) -> Dict[str, List[str]]:
        """Get information about numerical and categorical features"""
        return {
            'numerical': self.numerical_features,
            'categorical': self.categorical_features
        }

def create_preprocessor_ui(df: pd.DataFrame) -> PreprocessorConfig:
    """Create UI for configuring the preprocessor"""
    st.subheader("Preprocessor Configuration")
    
    with st.expander("Configure Preprocessor", expanded=True):
        # Column selection
        split_columns = st.multiselect(
            "Select columns to split",
            options=df.columns.tolist(),
            help="Select columns containing delimiter-separated key-value pairs"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            delimiter = st.text_input(
                "Delimiter",
                value="|",
                help="Character used to separate key-value pairs"
            )
            
            datetime_columns = st.multiselect(
                "DateTime columns",
                options=df.columns.tolist(),
                help="Select columns to convert to datetime"
            )
        
        with col2:
            key_value_separator = st.text_input(
                "Key-Value Separator",
                value="=",
                help="Character used to separate keys from values"
            )
            
            prefix_template = st.text_input(
                "Prefix Template",
                value="{column}_",
                help="Template for feature name prefix. Use {column} as placeholder"
            )
        
        output_dir = st.text_input(
            "Output Directory",
            value="processed_data",
            help="Directory to save processed data and configuration"
        )
        
        save_config = st.checkbox(
            "Save Configuration",
            value=True,
            help="Save configuration for future use"
        )
    
    return PreprocessorConfig(
        split_columns=split_columns,
        delimiter=delimiter,
        key_value_separator=key_value_separator,
        datetime_columns=datetime_columns,
        prefix_template=prefix_template,
        output_dir=output_dir,
        save_config=save_config
    )

def check_value_ranges(df: pd.DataFrame, columns_to_check: List[str], 
                      min_value: float = -100, max_value: float = 100) -> List[Dict]:
    """Check value ranges in specified columns of DataFrame."""
    def check_value_range(value):
        try:
            num_value = float(value)
            return min_value <= num_value <= max_value
        except (ValueError, TypeError):
            return True

    def process_cell(cell):
        """Process a cell value which might be numeric or string"""
        if pd.isna(cell) or cell == '':
            return []
            
        # If cell is numeric, check it directly
        if isinstance(cell, (int, float)):
            if not check_value_range(cell):
                return [('value', str(cell))]
            return []
            
        # If cell is string, try to split and process
        try:
            cell_str = str(cell)
            if '|' not in cell_str:
                # Single value without key-value structure
                if not check_value_range(cell_str):
                    return [('value', cell_str)]
                return []
                
            # Process key-value pairs
            pairs = cell_str.split('|')
            issues = []
            
            for pair in pairs:
                if '=' not in pair:
                    continue
                    
                name, value = pair.split('=')
                if not check_value_range(value):
                    issues.append((name.strip(), value.strip()))
            
            return issues
        except Exception as e:
            # If any error occurs, log it and continue
            print(f"Error processing cell {cell}: {str(e)}")
            return []

    issues = []
    
    for column in columns_to_check:
        if column not in df.columns:
            continue
            
        for idx, cell in df[column].items():
            cell_issues = process_cell(cell)
            
            for name, value in cell_issues:
                issues.append({
                    'row': idx + 2,  # Adding 2 for Excel row number (1-based + header)
                    'column': column,
                    'name': name,
                    'value': value
                })
    
    return issues

def run_value_range_check(file_path: str):
    """Run and display value range check results"""
    try:
        df = pd.read_csv(file_path)
        
        # Allow user to configure check
        st.subheader("Configure Value Range Check")
        with st.expander("Range Check Settings", expanded=True):
            columns_to_check = st.multiselect(
                "Select columns to check",
                options=df.columns.tolist(),
                default=['factors', 'score', 'efactors', 'exitScore'] if 'factors' in df.columns else None,
                help="Select columns containing key-value pairs to check"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                min_value = st.number_input("Minimum allowed value", value=-100.0)
            with col2:
                max_value = st.number_input("Maximum allowed value", value=100.0)
        
        if st.button("Run Check", type="primary"):
            issues = check_value_ranges(df, columns_to_check, min_value, max_value)
            
            if issues:
                st.warning(f"Found {len(issues)} values outside the range [{min_value}, {max_value}]:")
                
                issues_df = pd.DataFrame(issues)
                st.dataframe(issues_df)
                
                # Provide download option
                csv = issues_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"value_range_issues_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.success(f"No values found outside the range [{min_value}, {max_value}]")
    
    except Exception as e:
        st.error(f"Error checking value ranges: {str(e)}")


def preprocess_trading_data(file_path: str) -> None:
    """Enhanced preprocessing function with UI configuration"""
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Get configuration from UI
        config = create_preprocessor_ui(df)
        
        if st.button("Process Data", type="primary"):
            with st.spinner("Processing data..."):
                try:
                    # Create preprocessor with config
                    preprocessor = ConfigurableTradingPreprocessor(config)
                    
                    # Process data
                    processed_df = preprocessor.preprocess(df)
                    
                    # Save processed data
                    output_path = preprocessor.save_processed_data(processed_df, file_path)
                    
                    # Save configuration if enabled
                    if config.save_config:
                        preprocessor.save_configuration()
                    
                    # Display results
                    st.success(f"Data processed and saved to: {output_path}")
                    
                    # Create download button for processed data
                    csv = processed_df.to_csv(index=False)
                    st.download_button(
                        label="Download Processed Data",
                        data=csv,
                        file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Display processed data info
                    st.subheader("Processed Data Preview")
                    st.dataframe(processed_df.head())
                    
                    # Display feature information
                    features = preprocessor.get_feature_info()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Numerical Features:", len(features['numerical']))
                        st.write(features['numerical'])
                    
                    with col2:
                        st.write("Categorical Features:", len(features['categorical']))
                        st.write(features['categorical'])
                    
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

def scripts():
    """Main scripts page implementation"""
    # Add CSS for tooltip
    st.markdown("""
        <style>
        .header-container {
            display: flex;
            align-items: center;
            gap: 10px;
            position: relative;
        }
        .info-icon {
            color: #00ADB5;
            font-size: 1.2rem;
            cursor: help;
            text-decoration: none;
            position: relative;
            display: inline-block;
        }
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 500px;
            background-color: #252830;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 15px;
            position: absolute;
            z-index: 9999;
            top: -10px;
            left: 30px;
            opacity: 0;
            transition: opacity 0.3s;
            border: 1px solid #333;
            font-size: 0.9rem;
            line-height: 1.6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header with info icon
    st.markdown("""
        <div class='header-container'>
            <h2 style='color: #00ADB5; padding: 1rem 0; margin: 0;'>
                Scripts
            </h2>
            <div class='tooltip'>
                <span class='info-icon'>ℹ️</span>
                <div class='tooltiptext'>
                    All scripts here
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # File input
    file_path = st.text_input("Enter the path to your CSV file:", value="")
    
    if not file_path:
        st.info("Please enter a file path to proceed")
        return
        
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return
    
    # Create tabs for different tools
    tab1, tab2 = st.tabs(["Trading Data Preprocessor", "Value Range Checker"])
    
    with tab1:
        st.header("Trading Data Preprocessor")
        st.write("Preprocess trading data with configurable column splitting and feature extraction")
        preprocess_trading_data(file_path)
    
    with tab2:
        st.header("Value Range Checker")
        st.write("Check if values in specific columns are within defined ranges")
        run_value_range_check(file_path)

if __name__ == "__main__":
    scripts()