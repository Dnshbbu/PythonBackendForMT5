import streamlit as st
import io
import os
import chardet
import json

def load_saved_terms():
    """Load previously saved search terms."""
    try:
        with open('search_terms.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return ['Startup', 'Error', 'stopped']

def save_terms(terms):
    """Save search terms for future use."""
    with open('search_terms.json', 'w') as f:
        json.dump(terms, f)

def process_file(file_path, search_terms):
    """Process large files directly from disk."""
    results = {term: [] for term in search_terms}
    
    try:
        # First detect encoding from a sample
        with open(file_path, 'rb') as file:
            raw_data = file.read(min(10000, os.path.getsize(file_path)))
            encoding_result = chardet.detect(raw_data)
            encoding = encoding_result['encoding'] or 'utf-8'
        
        # Get total file size and calculate number of lines
        total_size = os.path.getsize(file_path)
        
        # Create a progress bar
        progress_bar = st.progress(0, "Analyzing log file...")
        
        # Process the file line by line
        line_buffer = {term: [] for term in search_terms}
        processed_size = 0
        
        with open(file_path, 'r', encoding=encoding, errors='replace') as file:
            for line in file:
                processed_size += len(line.encode(encoding))
                original_line = line.strip()
                
                if not original_line:
                    continue
                
                # Check each search term
                for term in search_terms:
                    # Add to buffer
                    line_buffer[term].append(original_line)
                    if len(line_buffer[term]) > 11:
                        line_buffer[term].pop(0)
                    
                    # Check if term is in line (case insensitive)
                    if term.lower() in original_line.lower():
                        context = {
                            'line': original_line,
                            'context': line_buffer[term].copy()
                        }
                        results[term].append(context)
                
                # Update progress based on file size
                if processed_size % 1000000 == 0:  # Update every ~1MB
                    progress = min(1.0, processed_size / total_size)
                    progress_bar.progress(progress)
        
        # Complete progress
        progress_bar.progress(1.0)
        progress_bar.empty()
        
        return results
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def display_results(results):
    """Display the results in the Streamlit interface"""
    # Use custom CSS to make the content wider and fix sidebar spacing
    st.markdown("""
        <style>
        <style>
        /* Reset and base styles */
        .stApp {
            overflow: auto !important;
        }

        /* Ensure main content container has proper overflow */
        .main .block-container {
            padding: 5rem 1rem 1rem !important;
            max-width: 100% !important;
            width: 100% !important;
            margin: 0 !important;
            overflow: visible !important;
            position: relative;
            z-index: 1;
        }

        /* Sidebar styling with lower z-index */
        [data-testid="stSidebar"] {
            height: 100vh !important;
            width: 24rem !important;
            position: fixed !important;
            overflow: auto !important;
            z-index: 99 !important;
        }

        /* Adjust header area z-index */
        .stApp > header {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            right: 0 !important;
            height: 40px !important;
            background-color: transparent !important;
            z-index: 100 !important;
            pointer-events: none !important;
        }

        /* Make header buttons clickable */
        .stApp > header button,
        .stApp > header a,
        [data-testid="stToolbar"],
        button[kind="header"] {
            pointer-events: auto !important;
        }

        /* Settings menu and toolbar */
        [data-testid="stToolbar"] {
            position: fixed !important;
            top: 0 !important;
            right: 1rem !important;
            height: 40px !important;
            padding-right: 1rem !important;
            z-index: 101 !important;
        }

        /* Ensure scrollbar is always accessible */
        ::-webkit-scrollbar {
            width: 12px !important;
            height: 12px !important;
            position: absolute !important;
            z-index: 102 !important;
            pointer-events: auto !important;
        }

        ::-webkit-scrollbar-track {
            background: #f1f1f1 !important;
            border-radius: 6px !important;
            margin: 2px;
        }

        ::-webkit-scrollbar-thumb {
            background: #888 !important;
            border-radius: 6px !important;
            border: 2px solid #f1f1f1;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #555 !important;
        }

        /* Firefox scrollbar */
        * {
            scrollbar-width: thin !important;
            scrollbar-color: #888 #f1f1f1 !important;
        }

        /* Create clickthrough zone for scrollbar */
        .main::after {
            content: '';
            position: fixed;
            top: 0;
            right: 0;
            width: 12px;
            height: 100vh;
            z-index: 98;
            pointer-events: auto !important;
        }

        /* Ensure header buttons remain clickable */
        button[kind="header"] {
            position: relative !important;
            z-index: 102 !important;
        }

        [data-testid="collapsedControl"] {
            display: block !important;
            position: fixed !important;
            top: 0.5rem !important;
            left: 1rem !important;
            z-index: 102 !important;
        }
        
        /* Sidebar configuration */
        section[data-testid="stSidebar"][aria-expanded="false"] {
            margin-left: -24rem;
        }
        
        section[data-testid="stSidebar"][aria-expanded="false"] + section.main {
            margin-left: 0;
        }
        
        section[data-testid="stSidebar"] {
            width: 24rem !important;
            min-width: 24rem !important;
        }
        
        /* Main container */
        .block-container {
            padding-top: 3rem !important;
            position: relative;
            z-index: 1;
            overflow: auto !important;
        }
        
        /* Make sure all containers allow scrolling */
        .element-container {
            overflow: visible !important;
        }
        
        .main .block-container {
            overflow: auto !important;
        }
        
        /* Explicitly show scrollbars */
        ::-webkit-scrollbar {
            width: 10px !important;
            height: 10px !important;
            display: block !important;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1 !important;
            border-radius: 4px !important;
            display: block !important;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #888 !important;
            border-radius: 4px !important;
            display: block !important;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #555 !important;
        }
        
        /* Ensure Firefox shows scrollbars */
        * {
            scrollbar-width: thin !important;
            scrollbar-color: #888 #f1f1f1 !important;
        }
        
        /* Code block styling */
        .stCodeBlock {
            position: relative;
            z-index: 1;
            user-select: text !important;
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        .stCodeBlock div {
            overflow-x: auto !important;
            max-height: 500px !important;
            overflow-y: auto !important;
            position: relative;
            z-index: 1;
        }
        
        pre {
            margin-top: 0 !important;
            padding-top: 0 !important;
            position: relative;
            z-index: 1;
        }
        
        .streamlit-expanderHeader {
            position: relative;
            z-index: 1;
        }

        /* Main container scrollbar */
        .main .block-container {
            overflow-y: auto;
            max-height: 100vh;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 14px;
            height: 14px;
            background-color: transparent;
            z-index: 999;
        }
        
        ::-webkit-scrollbar-track {
            background: #f0f0f0;
            border-radius: 7px;
            margin: 2px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #cdcdcd;
            border-radius: 7px;
            border: 3px solid #f0f0f0;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }
        
        /* Ensure scrollbars are always visible and interactive */
        .element-container, .stMarkdown, 
        div[data-testid="stExpander"] {
            overflow: visible !important;
        }
        
        /* Make code blocks scrollable */
        .stCodeBlock > div {
            max-height: 400px;
            overflow-y: auto !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Summary metrics
    st.subheader("📊 Search Results Summary")
    cols = st.columns(len(results))
    for col, (term, matches) in zip(cols, results.items()):
        col.metric(f"'{term}' found", len(matches))
    
    # Detailed results for each term
    for term, matches in results.items():
        with st.expander(f"🔍 Lines containing '{term}'", expanded=True):
            if not matches:
                st.info(f"No lines containing '{term}' found")
                continue
            
            for i, match in enumerate(matches, 1):
                st.text(f"Match {i}:")
                context_text = ""
                for context_line in match['context']:
                    prefix = ">" if context_line == match['line'] else " "
                    context_text += f"{prefix} {context_line}\n"
                st.code(context_text, language='text')

def main():
    # Set page configuration to wide mode
    st.set_page_config(
        page_title="MT5 Log File Analyzer",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Search Configuration")
        st.caption("Enter each search term on a new line")
        
        search_terms = st.text_area(
            "Search Terms",
            value="\n".join(load_saved_terms()),
            height=200,
            help="Enter the words you want to search for, one per line"
        ).split("\n")
        
        search_terms = [term.strip() for term in search_terms if term.strip()]
        
        if search_terms != load_saved_terms():
            save_terms(search_terms)
        
        st.markdown("---")
        st.caption("ℹ️ Tip: Click the '>' arrow at the top right to collapse/expand this sidebar")

    # Main content
    st.title("MT5 Log File Analyzer")
    
    # File path input
    file_path = st.text_input("Enter the path to your log file:", value="")
    
    if file_path:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
        else:
            # Show file details
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            st.info(f"File size: {file_size_mb:.2f} MB")
            
            if st.button("Analyze Log File", type="primary"):
                # Process the file
                results = process_file(file_path, search_terms)
                
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
                                    for context_line in match['context']:
                                        prefix = ">" if context_line == match['line'] else " "
                                        report.write(f"{prefix} {context_line}\n")
                            else:
                                report.write("No matches found\n")
                        
                        st.download_button(
                            label="Download Report",
                            data=report.getvalue(),
                            file_name="mt5_analysis_report.txt",
                            mime="text/plain"
                        )

if __name__ == "__main__":
    main()