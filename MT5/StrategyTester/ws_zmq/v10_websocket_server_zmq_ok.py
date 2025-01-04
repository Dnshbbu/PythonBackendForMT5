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

                            # If 'msg_content' contains data rows, write them
                            else:
                                # Append the data row
                                data_row = msg_content.split(',')
                                if len(data_row) == len(csv_headers):
                                    csv_writer.writerow(data_row)
                                    print(f"Appended data row to {csv_filename}: {data_row}")
                                else:
                                    print(f"Invalid data row format: {data_row}")

                    else:
                        print("Missing data fields in the incoming JSON.")

            except zmq.Again:
                # No message received, just pass
                pass

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
