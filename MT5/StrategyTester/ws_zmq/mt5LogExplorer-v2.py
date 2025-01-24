import streamlit as st
import io
import os
import chardet
import json
import pandas as pd

# Keep all your existing utility functions from mt5LogExplorer.py
def load_saved_settings():
    try:
        with open('analyzer_settings.json', 'r') as f:
            settings = json.load(f)
            return (
                settings.get('terms', ['Startup', 'Error', 'stopped']),
                settings.get('lines_before', 5),
                settings.get('lines_after', 5)
            )
    except FileNotFoundError:
        return ['Startup', 'Error', 'stopped'], 5, 5

def save_settings(terms, lines_before, lines_after):
    settings = {
        'terms': terms,
        'lines_before': lines_before,
        'lines_after': lines_after
    }
    with open('analyzer_settings.json', 'w') as f:
        json.dump(settings, f)

def process_file(file_path, search_terms, lines_before, lines_after):
    # Your existing process_file function code here
    results = {term: [] for term in search_terms}
    
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read(min(10000, os.path.getsize(file_path)))
            encoding_result = chardet.detect(raw_data)
            encoding = encoding_result['encoding'] or 'utf-8'
        
        total_size = os.path.getsize(file_path)
        progress_bar = st.progress(0, "Analyzing log file...")
        
        with open(file_path, 'r', encoding=encoding, errors='replace') as file:
            all_lines = file.readlines()
            total_lines = len(all_lines)
            
            for current_line_idx, line in enumerate(all_lines):
                processed_size = file.tell()
                original_line = line.strip()
                
                if not original_line:
                    continue
                
                for term in search_terms:
                    if term.lower() in original_line.lower():
                        start_idx = max(0, current_line_idx - lines_before)
                        end_idx = min(total_lines, current_line_idx + lines_after + 1)
                        
                        context_lines = [l.strip() for l in all_lines[start_idx:end_idx]]
                        match_index = current_line_idx - start_idx
                        
                        results[term].append({
                            'line': original_line,
                            'context': context_lines,
                            'match_index': match_index
                        })
                
                if processed_size % 1000000 == 0:
                    progress = min(1.0, processed_size / total_size)
                    progress_bar.progress(progress)
        
        progress_bar.progress(1.0)
        progress_bar.empty()
        
        return results
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def display_results(results):
    st.subheader("📊 Search Results Summary")
    cols = st.columns(len(results))
    for col, (term, matches) in zip(cols, results.items()):
        col.metric(f"'{term}' found", len(matches))
    
    for term, matches in results.items():
        with st.expander(f"🔍 Lines containing '{term}'", expanded=False):
            if not matches:
                st.info(f"No lines containing '{term}' found")
                continue
            
            for i, match in enumerate(matches, 1):
                st.text(f"Match {i}:")
                context_text = ""
                for idx, context_line in enumerate(match['context']):
                    prefix = ">" if idx == match['match_index'] else " "
                    context_text += f"{prefix} {context_line}\n"
                st.code(context_text, language='text')

def check_value_ranges(file_path: str):
    """Check value ranges in CSV file"""
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

def log_explorer():
    st.title("MT5 Log File Analyzer")
    
    # Load saved settings
    saved_terms, saved_lines_before, saved_lines_after = load_saved_settings()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Search Configuration")
        st.caption("Enter each search term on a new line")
        
        search_terms = st.text_area(
            "Search Terms",
            value="\n".join(saved_terms),
            height=200,
            help="Enter the words you want to search for, one per line"
        ).split("\n")
        
        search_terms = [term.strip() for term in search_terms if term.strip()]
        
        st.markdown("---")
        
        st.subheader("Context Configuration")
        lines_before = st.number_input(
            "Lines before match",
            min_value=0,
            max_value=50,
            value=saved_lines_before
        )
        
        lines_after = st.number_input(
            "Lines after match",
            min_value=0,
            max_value=50,
            value=saved_lines_after
        )
        
        if (search_terms != saved_terms or 
            lines_before != saved_lines_before or 
            lines_after != saved_lines_after):
            save_settings(search_terms, lines_before, lines_after)
    
    file_path = st.text_input("Enter the path to your log file:", value="")
    
    if file_path and os.path.exists(file_path):
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        st.info(f"File size: {file_size_mb:.2f} MB")
        
        if st.button("Analyze Log File", type="primary"):
            results = process_file(file_path, search_terms, lines_before, lines_after)
            if results:
                display_results(results)

def scripts():
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
            
            # Add export option
            csv = issues_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="value_range_issues.csv",
                mime="text/csv"
            )
        else:
            st.success("No values found outside the range -100 to +100")

def main():
    st.set_page_config(
        page_title="MT5 Log File Analyzer",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Simple page navigation using selectbox in the main area
    pages = {
        "Log Explorer": log_explorer,
        "Scripts": scripts
    }
    
    # Add the navigation to the sidebar
    page = st.sidebar.selectbox("Navigation", list(pages.keys()))
    
    # Call the selected page function
    pages[page]()

if __name__ == "__main__":
    main()