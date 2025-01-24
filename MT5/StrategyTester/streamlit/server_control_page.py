# server_control_page.py
import streamlit as st
import subprocess
import os
import psutil
import sys
from typing import Tuple, Optional

class ServerController:
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.server_script = os.path.join(self.current_dir, 'zmqServer.py')

    def start_powershell_server(self) -> Tuple[bool, str]:
        """Start ZMQ server in a new PowerShell window"""
        try:
            if not os.path.exists(self.server_script):
                return False, f"Server script not found at: {self.server_script}"

            powershell_command = [
                'powershell.exe',
                'Start-Process',
                'powershell',
                '-ArgumentList',
                f'"-NoExit -Command python \'{self.server_script}\'"'
            ]
            
            subprocess.Popen(powershell_command, creationflags=subprocess.CREATE_NEW_CONSOLE)
            return True, "Server started in new PowerShell window"
        except Exception as e:
            return False, f"Error starting server: {str(e)}"

    def get_server_status(self) -> Tuple[bool, Optional[int]]:
        """Check if ZMQ server is running"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline'] and len(proc.info['cmdline']) > 1:
                    if 'python' in proc.info['cmdline'][0].lower() and 'zmqserver.py' in proc.info['cmdline'][1].lower():
                        return True, proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return False, None

def server_control_page():
    """Server Control page implementation"""
    st.title("MT5 Server Control Panel")
    
    controller = ServerController()
    is_running, pid = controller.get_server_status()

    # Status display
    if is_running:
        st.success(f"Server Status: Running (PID: {pid})")
    else:
        st.warning("Server Status: Stopped")

    # Control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if not is_running and st.button("Start Server in PowerShell", type="primary"):
            success, message = controller.start_powershell_server()
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    
    with col2:
        if st.button("Check Server Status", type="secondary"):
            st.rerun()

    # Server logs section
    st.header("Server Logs")
    log_file = os.path.join(controller.current_dir, "mt5_zmq_client.log")
    
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = f.readlines()
            
            num_lines = st.slider("Number of log lines to show", 10, 200, 50)
            st.code("".join(logs[-num_lines:]), language="plaintext")
            
            if st.button("Refresh Logs"):
                st.rerun()
        else:
            st.info("No log file found")
    except Exception as e:
        st.error(f"Error reading logs: {str(e)}")