# scripts_page.py
import streamlit as st
import pandas as pd
import os

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