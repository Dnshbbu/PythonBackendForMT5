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
                ZMQ Server Control Panel
            </h2>
            <div class='tooltip'>
                <span class='info-icon'>ℹ️</span>
                <div class='tooltiptext'>
                    We are controlling the start/stop of zmq server
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

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