import asyncio
import zmq
import zmq.asyncio
import csv
import json
import os
from datetime import datetime

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

    print("Asynchronous Python client started. Waiting for MT5 tick data...")

    try:
        while True:
            try:
                # Try to receive tick data asynchronously
                message = await socket_receive.recv_string()
                # Check if the message is not empty
                if message:
                    try:
                        # Parse the JSON message
                        data = json.loads(message)
                    except json.JSONDecodeError as e:
                        print(f"[{datetime.now()}] JSONDecodeError: {e}. Message: {message}")
                        # Optionally, you can log the error to a file or take other actions
                        continue  # Skip processing this message and continue with the next
                    
                    run_id = data.get("run_id")
                    msg_type = data.get("type")
                    msg_content = data.get("msg")

                    if run_id and msg_type and msg_content:
                        # Generate the CSV filename
                        csv_filename = f"{run_id}_{msg_type}.csv"
                        csv_headers = msg_content.split(',')

                        # Check if the file already exists
                        file_exists = os.path.isfile(csv_filename)

                        # Open the file in append mode
                        with open(csv_filename, 'a', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            
                            # If the file does not exist, write the headers
                            if not file_exists:
                                csv_writer.writerow(csv_headers)
                                print(f"Created CSV file: {csv_filename} with headers: {csv_headers}")

                            # Write the data rows
                            # Assuming that additional data messages contain data corresponding to the headers
                            # Here, we need to define how the data is received.
                            # For demonstration, let's assume 'data_rows' is a list of comma-separated strings
                            # You might need to adjust this based on your actual message structure.

                            # Example: If 'msg_content' contains data rows instead of headers, adjust accordingly
                            # For now, assuming headers are sent once and data comes separately
                            # You might need to handle differently based on your EA's message structure

                            # If 'msg_content' contains data rows, uncomment the following lines:
                            # data_rows = msg_content.split(';')
                            # for row in data_rows:
                            #     row_data = row.split(',')
                            #     if len(row_data) == len(csv_headers):
                            #         csv_writer.writerow(row_data)
                            #     else:
                            #         print(f"Invalid data row format: {row}")

                            # However, based on your initial message, it seems 'msg' contains headers.
                            # If data rows are sent in separate messages, you need to handle them accordingly.

                    else:
                        print("Missing data fields in the incoming JSON.")

            except zmq.Again:
                # No message received, just pass
                pass
                # print("No data received yet...")

            # Small delay to prevent tight loop
            await asyncio.sleep(0.01)

    except asyncio.CancelledError:
        print("Stopping Python client...")

    finally:
        # Clean up
        socket_send.close()
        socket_receive.close()
        context.term()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting...")
