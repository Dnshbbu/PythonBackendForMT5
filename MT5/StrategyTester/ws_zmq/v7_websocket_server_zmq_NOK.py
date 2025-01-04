import asyncio
import zmq
import zmq.asyncio
import csv
import json
import os
from datetime import datetime
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("client.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Fix for Proactor Event Loop Warning on Windows
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def main():
    context = zmq.asyncio.Context()

    # Socket to send commands
    socket_send = context.socket(zmq.PUSH)
    socket_send.connect("tcp://127.0.0.1:5557")  # Connect to the PUSH socket of the EA

    # Socket to receive responses
    socket_receive = context.socket(zmq.PULL)
    socket_receive.connect("tcp://127.0.0.1:5556")  # Connect to the PULL socket of the EA

    logging.info("Asynchronous Python client started. Waiting for MT5 tick data...")

    # Dictionary to keep track of CSV headers for each (run_id, msg_type)
    csv_headers_dict = {}

    try:
        while True:
            try:
                # Try to receive tick data asynchronously
                message = await socket_receive.recv_string()

                # Debug: Print the raw message
                logging.debug(f"Received raw message: {message}")

                if not message:
                    logging.warning("Received an empty message.")
                    continue

                # Attempt to parse the message as JSON
                try:
                    data = json.loads(message)
                    if not isinstance(data, dict):
                        logging.error(f"JSON decoded but not a dict: {data}")
                        continue

                    run_id = data.get("run_id")
                    msg_type = data.get("type")
                    msg_content = data.get("msg")

                    if not all([run_id, msg_type, msg_content]):
                        logging.warning(f"Missing fields in JSON message: {data}")
                        continue

                    # Generate the CSV filename
                    csv_filename = f"{run_id}_{msg_type}.csv"

                    if msg_type == "PriceAndEquity":
                        # Initialize headers if not already done
                        if (run_id, msg_type) not in csv_headers_dict:
                            csv_headers = msg_content.split(',')
                            csv_headers_dict[(run_id, msg_type)] = csv_headers

                            # Check if the file already exists
                            file_exists = os.path.isfile(csv_filename)

                            with open(csv_filename, 'a', newline='') as csvfile:
                                csv_writer = csv.writer(csvfile)

                                # If the file does not exist, write the headers
                                if not file_exists:
                                    csv_writer.writerow(csv_headers)
                                    # logging.info(f"Created CSV file: {csv_filename} with headers: {csv_headers}")

                        # Handle data rows
                        csv_headers = csv_headers_dict.get((run_id, msg_type))
                        if not csv_headers:
                            logging.error(f"No headers found for run_id: {run_id}, msg_type: {msg_type}")
                            continue  # Skip if headers are missing

                        # Split data rows based on ';'
                        data_rows = msg_content.split(';')
                        with open(csv_filename, 'a', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            for row in data_rows:
                                row = row.strip()
                                if not row:
                                    continue  # Skip empty rows
                                row_data = row.split(',')
                                if len(row_data) == len(csv_headers):
                                    csv_writer.writerow(row_data)
                                else:
                                    logging.warning(f"Invalid data row format in {csv_filename}: {row}")

                    elif msg_type == "transactions":
                        # Just append the message content to the CSV file without splitting
                        with open(csv_filename, 'a', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerow([msg_content])
                            # logging.info(f"Appended transaction data to {csv_filename}: {msg_content}")

                    else:
                        logging.warning(f"Unhandled message type: {msg_type}")

                except json.JSONDecodeError:
                    # If message is not JSON, attempt to handle as raw data
                    logging.debug("Message is not JSON. Attempting to process as raw data.")

                    # Attempt to process as raw CSV-like data
                    # Assuming raw data follows a specific format, e.g., "field1,field2,...;field1,field2,..."
                    try:
                        data_rows = message.split(';')
                        if not data_rows:
                            logging.warning("No data rows found in raw message.")
                            continue

                        # For raw data, you might need to define how to handle it.
                        # For example, assume a default run_id and msg_type
                        default_run_id = "default_run"
                        default_msg_type = "raw_data"
                        csv_filename = f"{default_run_id}_{default_msg_type}.csv"

                        # Initialize headers if not already done
                        if (default_run_id, default_msg_type) not in csv_headers_dict:
                            # Assuming the first raw message contains headers
                            csv_headers = data_rows[0].split(',')
                            csv_headers_dict[(default_run_id, default_msg_type)] = csv_headers

                            # Check if the file already exists
                            file_exists = os.path.isfile(csv_filename)

                            with open(csv_filename, 'a', newline='') as csvfile:
                                csv_writer = csv.writer(csvfile)

                                # If the file does not exist, write the headers
                                if not file_exists:
                                    csv_writer.writerow(csv_headers)
                                    # logging.info(f"Created CSV file: {csv_filename} with headers: {csv_headers}")
                            
                            # Remove the header row from data_rows
                            data_rows = data_rows[1:]

                        # Now append the data rows
                        csv_headers = csv_headers_dict.get((default_run_id, default_msg_type))
                        with open(csv_filename, 'a', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            for row in data_rows:
                                row = row.strip()
                                if not row:
                                    continue  # Skip empty rows
                                row_data = row.split(',')
                                if len(row_data) == len(csv_headers):
                                    csv_writer.writerow(row_data)
                                else:
                                    logging.warning(f"Invalid raw data row format in {csv_filename}: {row}")

                    except Exception as e:
                        logging.error(f"Failed to process raw message: {message}. Error: {e}")
                        continue  # Skip to the next message

            except zmq.Again:
                # No message received, just pass
                pass
                # logging.debug("No data received yet...")

            except Exception as e:
                # Catch any other unexpected exceptions to prevent the program from stopping
                logging.error(f"An unexpected error occurred while processing message: {e}", exc_info=True)

            # Small delay to prevent tight loop
            await asyncio.sleep(0.01)

    except asyncio.CancelledError:
        logging.info("Stopping Python client...")

    finally:
        # Clean up
        socket_send.close()
        socket_receive.close()
        context.term()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Exiting...")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
