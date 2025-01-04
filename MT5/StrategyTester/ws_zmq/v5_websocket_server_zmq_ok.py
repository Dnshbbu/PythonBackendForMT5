import asyncio
import zmq
import zmq.asyncio
import csv
from datetime import datetime

async def main():
    context = zmq.asyncio.Context()

    # Socket to send commands
    socket_send = context.socket(zmq.PUSH)
    socket_send.connect("tcp://127.0.0.1:5557")  # Connect to the PUSH socket of the EA

    # Socket to receive responses
    socket_receive = context.socket(zmq.PULL)
    socket_receive.connect("tcp://127.0.0.1:5556")  # Connect to the PULL socket of the EA

    print("Asynchronous Python client started. Waiting for MT5 tick data...")

    # Set up CSV file
    csv_filename = f"tick_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_headers = ['Date', 'Time', 'Symbol', 'Equity', 'Price']

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_headers)

    try:
        while True:
            try:
                # Try to receive tick data asynchronously
                message = await socket_receive.recv_string()
                # Check if the message is not empty
                if message:
                    # Split the batched message
                    tick_data_list = message.split(';')
                    
                    with open(csv_filename, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        
                        for tick_data_str in tick_data_list:
                            # Split each tick data into its components
                            tick_components = tick_data_str.split(',')
                            
                            if len(tick_components) == 5:
                                # Write to CSV
                                csv_writer.writerow(tick_components)
                                
                                # Print to console (optional)
                                #print(f"Received: {tick_data_str}")
                            else:
                                print(f"Invalid tick data format: {tick_data_str}")

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