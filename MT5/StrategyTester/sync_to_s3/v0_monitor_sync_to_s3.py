import os
import time
import hashlib
import boto3
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

# Function to check if a file is stable (i.e., finished writing)
def is_file_stable(file_path, wait_time=5):
    """Returns True if the file's size has not changed during the wait_time."""
    initial_size = os.path.getsize(file_path)
    time.sleep(wait_time)
    final_size = os.path.getsize(file_path)
    return initial_size == final_size

# Function to list all files in the S3 bucket
def list_s3_files(bucket_name):
    s3_files = {}
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name):
        if 'Contents' in page:
            for obj in page['Contents']:
                s3_files[obj['Key']] = obj['ETag'].strip('"')  # ETag is the MD5 checksum
    return s3_files

# Event handler for monitoring file system changes
class S3SyncEventHandler(FileSystemEventHandler):
    def __init__(self, local_directory, bucket_name):
        self.local_directory = local_directory
        self.bucket_name = bucket_name
        self.s3_files = list_s3_files(bucket_name)

    def on_modified(self, event):
        if not event.is_directory:
            self.upload_to_s3(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self.upload_to_s3(event.src_path)

    def upload_to_s3(self, local_file_path):
        relative_file_path = os.path.relpath(local_file_path, self.local_directory).replace("\\", "/")

        # Check if the file is stable (finished being written)
        if not is_file_stable(local_file_path):
            print(f"File {relative_file_path} is still being written. Skipping.")
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
    local_directory = r'C:\Users\StdUser\Desktop\MyProjects\Backtesting\test'
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
