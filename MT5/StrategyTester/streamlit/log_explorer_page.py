# log_explorer_page.py
import streamlit as st
import io
import os
import chardet
import json

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