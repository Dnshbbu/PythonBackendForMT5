import os
import time
import hashlib
import boto3
import psutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

s3 = boto3.client('s3')

# Helper function to calculate the MD5 checksum of a file
def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Function to check if a file is open (using psutil)
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

# Function to list all files in the S3 bucket
def list_s3_files(bucket_name):
    s3_files = {}
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name):
        if 'Contents' in page:
            for obj in page['Contents']:
                s3_files[obj['Key']] = obj['ETag'].strip('"')  # ETag is the MD5 checksum
    return s3_files

# Function to sync existing files at startup
def sync_existing_files(local_directory, bucket_name, s3_files):
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

            if relative_file_path not in s3_files:
                print(f"Uploading {relative_file_path} to S3 (existing file)")
                s3.upload_file(local_file_path, bucket_name, relative_file_path)
                s3_files[relative_file_path] = local_md5
            elif s3_files[relative_file_path] != local_md5:
                print(f"File {relative_file_path} exists but differs from S3 version. Uploading updated version (existing file).")
                s3.upload_file(local_file_path, bucket_name, relative_file_path)
                s3_files[relative_file_path] = local_md5
            else:
                print(f"File {relative_file_path} is already up-to-date in S3 (existing file).")

# Event handler for monitoring file system changes
class S3SyncEventHandler(FileSystemEventHandler):
    def __init__(self, local_directory, bucket_name):
        self.local_directory = local_directory
        self.bucket_name = bucket_name
        self.s3_files = list_s3_files(bucket_name)

        # Sync existing files at startup
        sync_existing_files(local_directory, bucket_name, self.s3_files)

    def on_modified(self, event):
        if not event.is_directory:
            self.upload_to_s3(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self.upload_to_s3(event.src_path)

    def upload_to_s3(self, local_file_path):
        relative_file_path = os.path.relpath(local_file_path, self.local_directory).replace("\\", "/")

        # Check if the file is still being written or in use (using psutil)
        if is_file_open(local_file_path):
            print(f"File {relative_file_path} is still being written or in use. Skipping.")
            return

        local_md5 = calculate_md5(local_file_path)

        if relative_file_path not in self.s3_files:
            print(f"Uploading {relative_file_path} to S3")
            s3.upload_file(local_file_path, self.bucket_name, relative_file_path)
            self.s3_files[relative_file_path] = local_md5
        elif self.s3_files[relative_file_path] != local_md5:
            print(f"File {relative_file_path} exists but differs from S3 version. Uploading updated version.")
            s3.upload_file(local_file_path, self.bucket_name, relative_file_path)
            self.s3_files[relative_file_path] = local_md5
        else:
            print(f"File {relative_file_path} is already up-to-date in S3.")

if __name__ == "__main__":
 # Set your local directory and S3 bucket name
    local_directory = r'C:\Users\StdUser\Desktop\MyProjects\Backtesting\logs'
    bucket_name = 'strategytester'

    # Create an event handler and observer
    event_handler = S3SyncEventHandler(local_directory, bucket_name)
    observer = Observer()
    observer.schedule(event_handler, local_directory, recursive=True)

    # Start monitoring
    observer.start()
    print(f"Monitoring {local_directory} for changes...")

    try:
        while True:
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
