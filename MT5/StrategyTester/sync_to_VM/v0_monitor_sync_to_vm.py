import os
import time
import hashlib
import paramiko
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Helper function to calculate the MD5 checksum of a file
def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Function to check if a file is open (using psutil)
import psutil
def is_file_open(file_path):
    """Check if a file is currently open by any process."""
    for proc in psutil.process_iter(['open_files']):
        try:
            open_files = proc.info['open_files']
            if open_files:
                for open_file in open_files:
                    if open_file.path == file_path:
                        return True  # File is open
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

# Function to connect via SSH
def create_ssh_client(hostname, username, key_filepath):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, username=username, key_filename=key_filepath)
    return ssh

# Function to upload files via SCP (using SSH)
def upload_file_via_scp(local_file_path, remote_directory, ssh_client):
    scp = ssh_client.open_sftp()
    remote_file_path = os.path.join(remote_directory, os.path.basename(local_file_path)).replace("\\", "/")
    scp.put(local_file_path, remote_file_path)
    scp.close()

# Sync existing files at startup
def sync_existing_files(local_directory, remote_directory, ssh_client, synced_files):
    print("Checking existing files in the directory for sync...")

    for root, _, files in os.walk(local_directory):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            relative_file_path = os.path.relpath(local_file_path, local_directory).replace("\\", "/")

            # Check if the file is being written or in use
            if is_file_open(local_file_path):
                print(f"File {relative_file_path} is still being written or in use. Skipping.")
                continue

            local_md5 = calculate_md5(local_file_path)

            if relative_file_path not in synced_files:
                print(f"Uploading {relative_file_path} to VM (existing file)")
                upload_file_via_scp(local_file_path, remote_directory, ssh_client)
                synced_files[relative_file_path] = local_md5
            elif synced_files[relative_file_path] != local_md5:
                print(f"File {relative_file_path} differs from VM version. Uploading updated version.")
                upload_file_via_scp(local_file_path, remote_directory, ssh_client)
                synced_files[relative_file_path] = local_md5
            else:
                print(f"File {relative_file_path} is already up-to-date on VM.")

# Event handler for monitoring file system changes
class VMFileSyncEventHandler(FileSystemEventHandler):
    def __init__(self, local_directory, remote_directory, ssh_client):
        self.local_directory = local_directory
        self.remote_directory = remote_directory
        self.ssh_client = ssh_client
        self.synced_files = {}

        # Sync existing files at startup
        sync_existing_files(local_directory, remote_directory, ssh_client, self.synced_files)

    def on_modified(self, event):
        if not event.is_directory:
            self.upload_to_vm(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self.upload_to_vm(event.src_path)

    def upload_to_vm(self, local_file_path):
        relative_file_path = os.path.relpath(local_file_path, self.local_directory).replace("\\", "/")

        # Check if the file is still being written or in use (using psutil)
        if is_file_open(local_file_path):
            print(f"File {relative_file_path} is still being written or in use. Skipping.")
            return

        local_md5 = calculate_md5(local_file_path)

        if relative_file_path not in self.synced_files:
            print(f"Uploading {relative_file_path} to VM")
            upload_file_via_scp(local_file_path, self.remote_directory, self.ssh_client)
            self.synced_files[relative_file_path] = local_md5
        elif self.synced_files[relative_file_path] != local_md5:
            print(f"File {relative_file_path} differs from VM version. Uploading updated version.")
            upload_file_via_scp(local_file_path, self.remote_directory, self.ssh_client)
            self.synced_files[relative_file_path] = local_md5
        else:
            print(f"File {relative_file_path} is already up-to-date on VM.")

if __name__ == "__main__":
    # Set your local directory and remote VM details
    local_directory = r'C:\Users\StdUser\Desktop\MyProjects\Backtesting\logs'
    remote_directory = '/home/ubuntu/serve-files'
    hostname = '143.47.237.191'
    username = 'ubuntu'
    key_filepath = r'C:\Users\StdUser\Documents\oracle_free_tier\ssh-key-2024-10-18.key'

    # Create an SSH client connection
    ssh_client = create_ssh_client(hostname, username, key_filepath)

    # Create an event handler and observer
    event_handler = VMFileSyncEventHandler(local_directory, remote_directory, ssh_client)
    observer = Observer()
    observer.schedule(event_handler, local_directory, recursive=True)

    # Start monitoring
    observer.start()
    print(f"Monitoring {local_directory} for changes and syncing to {hostname}:{remote_directory}...")

    try:
        while True:
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
    ssh_client.close()
