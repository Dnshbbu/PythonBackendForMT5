# zmq_server_page.py
import streamlit as st
import os
import psutil
import subprocess
import sys
import time

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