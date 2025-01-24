import streamlit as st
import io
import os
import chardet
import json
import pandas as pd
import numpy as np
import subprocess
import sys
import psutil
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Utility functions for Log Explorer
def load_saved_settings():
    """Load previously saved search terms and context settings."""
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
    """Save search terms and context settings for future use."""
    settings = {
        'terms': terms,
        'lines_before': lines_before,
        'lines_after': lines_after
    }
    with open('analyzer_settings.json', 'w') as f:
        json.dump(settings, f)

def process_file(file_path, search_terms, lines_before, lines_after):
    """Process log file and search for terms."""
    results = {term: [] for term in search_terms}
    
    try:
        # Detect encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read(min(10000, os.path.getsize(file_path)))
            encoding_result = chardet.detect(raw_data)
            encoding = encoding_result['encoding'] or 'utf-8'
        
        # Create progress bar
        progress_bar = st.progress(0, "Analyzing log file...")
        total_size = os.path.getsize(file_path)
        processed_size = 0
        
        with open(file_path, 'r', encoding=encoding, errors='replace') as file:
            all_lines = file.readlines()
            total_lines = len(all_lines)
            
            for current_line_idx, line in enumerate(all_lines):
                processed_size += len(line.encode(encoding))
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
                
                if processed_size % 1000000 == 0:  # Update every ~1MB
                    progress = min(1.0, processed_size / total_size)
                    progress_bar.progress(progress)
        
        progress_bar.progress(1.0)
        progress_bar.empty()
        
        return results
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def display_results(results):
    """Display search results in the Streamlit interface."""
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

# Scripts Functions
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

# Server Control Functions
def get_server_status():
    """Check if the ZMQ server is running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline'] and len(proc.info['cmdline']) > 1:
                if 'python' in proc.info['cmdline'][0].lower() and 'zmqserver.py' in proc.info['cmdline'][1].lower():
                    return True, proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False, None

def start_server():
    """Start the ZMQ server"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        server_path = os.path.join(current_dir, 'zmqServer.py')
        
        if not os.path.exists(server_path):
            return False, f"Server file not found at: {server_path}"
        
        process = subprocess.Popen(
            [sys.executable, server_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        
        time.sleep(2)
        
        if process.poll() is None:  # Process is still running
            return True, "Server started successfully"
        else:
            return False, f"Server failed to start: {process.stderr.read().decode()}"
    except Exception as e:
        return False, f"Error starting server: {str(e)}"

def stop_server(pid):
    """Stop the ZMQ server"""
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(['taskkill', '/F', '/PID', str(pid)], check=True)
        else:  # Linux/Mac
            subprocess.run(['kill', '-9', str(pid)], check=True)
        return True, "Server stopped successfully"
    except Exception as e:
        return False, f"Error stopping server: {str(e)}"

# Machine Learning Functions
def run_clustering(df, algorithm, params=None):
    """
    Run clustering algorithms on the provided DataFrame.
    
    Args:
        df: pandas DataFrame
        algorithm: str, name of the clustering algorithm
        params: dict, algorithm parameters (optional)
    """
    # Prepare the data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return None, "No numeric columns found in the dataset"
    
    X = df[numeric_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize the algorithm
    if params is None:
        params = {}
        
    try:
        if algorithm == "kmeans":
            model = KMeans(n_clusters=params.get("n_clusters", 3), random_state=42)
        elif algorithm == "dbscan":
            model = DBSCAN(eps=params.get("eps", 0.5), min_samples=params.get("min_samples", 5))
        elif algorithm == "hierarchical":
            model = AgglomerativeClustering(n_clusters=params.get("n_clusters", 3))
        else:
            return None, f"Unknown algorithm: {algorithm}"
            
        # Fit the model
        labels = model.fit_predict(X_scaled)
        
        # Add cluster labels to the original dataframe
        result_df = df.copy()
        result_df['Cluster'] = labels
        
        # Calculate basic statistics
        stats = {
            'n_clusters': len(np.unique(labels)),
            'cluster_sizes': pd.Series(labels).value_counts().to_dict()
        }
        
        return result_df, stats
        
    except Exception as e:
        return None, str(e)

def run_regression(df, algorithm, target_column, params=None):
    """
    Run regression algorithms on the provided DataFrame.
    
    Args:
        df: pandas DataFrame
        algorithm: str, name of the regression algorithm
        target_column: str, name of the target column
        params: dict, algorithm parameters (optional)
    """
    if target_column not in df.columns:
        return None, f"Target column '{target_column}' not found in dataset"
    
    # Prepare the data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols[numeric_cols != target_column]
    
    if len(numeric_cols) == 0:
        return None, "No numeric features found in the dataset"
    
    X = df[numeric_cols]
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize the algorithm
    if params is None:
        params = {}
        
    try:
        if algorithm == "linear":
            model = LinearRegression()
        elif algorithm == "ridge":
            model = Ridge(alpha=params.get("alpha", 1.0))
        elif algorithm == "lasso":
            model = Lasso(alpha=params.get("alpha", 1.0))
        else:
            return None, f"Unknown algorithm: {algorithm}"
            
        # Fit and predict
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': numeric_cols,
            'importance': np.abs(model.coef_)
        }).sort_values('importance', ascending=False)
        
        return metrics, feature_importance
        
    except Exception as e:
        return None, str(e)

# Page Implementations
def log_explorer():
    """Log Explorer page implementation"""
    st.title("MT5 Log File Analyzer")
    
    saved_terms, saved_lines_before, saved_lines_after = load_saved_settings()
    
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
                
                # Add export option
                if st.button("Export Results"):
                    report = io.StringIO()
                    report.write("=== MT5 Log Analysis Report ===\n\n")
                    
                    for term, matches in results.items():
                        report.write(f"\n=== Lines containing '{term}' ===\n")
                        if matches:
                            for i, match in enumerate(matches, 1):
                                report.write(f"\nMatch {i}:\n")
                                for idx, context_line in enumerate(match['context']):
                                    prefix = ">" if idx == match['match_index'] else " "
                                    report.write(f"{prefix} {context_line}\n")
                        else:
                            report.write("No matches found\n")
                    
                    st.download_button(
                                            label="Download Report",
                                            data=report.getvalue(),
                                            file_name="mt5_analysis_report.txt",
                                            mime="text/plain"
                                        )
                elif file_path:
                    st.error(f"File not found: {file_path}")

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

def zmq_server():
    """ZMQ Server Control page implementation"""
    st.title("MT5 ZMQ Server Control")
    
    # Check server status
    is_running, pid = get_server_status()
    
    # Display server status
    status_container = st.container()
    with status_container:
        if is_running:
            st.success(f"Server is running (PID: {pid})")
        else:
            st.warning("Server is not running")
    
    # Server control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if not is_running:
            if st.button("Start Server", type="primary"):
                success, message = start_server()
                if success:
                    st.success(message)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(message)
    
    with col2:
        if is_running:
            if st.button("Stop Server", type="secondary"):
                success, message = stop_server(pid)
                if success:
                    st.success(message)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(message)
    
    # Server logs section
    st.header("Server Logs")
    try:
        log_file = "mt5_zmq_client.log"
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = f.readlines()
            
            # Show last 50 lines by default
            num_lines = st.slider("Number of log lines to show", 10, 200, 50)
            logs = logs[-num_lines:]
            
            # Display logs in a scrollable text area
            log_text = "".join(logs)
            st.code(log_text, language="plaintext")
            
            # Add refresh button
            if st.button("Refresh Logs"):
                st.rerun()
        else:
            st.info("No log file found")
    except Exception as e:
        st.error(f"Error reading log file: {str(e)}")


def main():
    st.set_page_config(
        page_title="MT5 Analysis Tools",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Page navigation
    pages = {
        "Log Explorer": log_explorer,
        "Scripts": scripts,
        "ZMQ Server": zmq_server,
        "ML Analysis": sklearn_page
    }
    
    # Add the navigation to the sidebar
    page = st.sidebar.selectbox("Navigation", list(pages.keys()))
    
    # Call the selected page function
    pages[page]()

if __name__ == "__main__":
    main()