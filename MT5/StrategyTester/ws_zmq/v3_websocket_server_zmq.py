import zmq
import time
import json

context = zmq.Context()

# Socket to send commands
socket_send = context.socket(zmq.PUSH)
socket_send.connect("tcp://127.0.0.1:5557")  # Connect to the PUSH socket of the EA

# Socket to receive responses
socket_receive = context.socket(zmq.PULL)
socket_receive.connect("tcp://127.0.0.1:5556")  # Connect to the PULL socket of the EA

print("Python client started. Waiting for MT5 tick data...")

try:
    while True:
        try:
            # Try to receive tick data
            message = socket_receive.recv_string(flags=zmq.NOBLOCK)
            # Check if the message is not empty
            if message:
                try:
                    # Parse the JSON-like string
                    tick_data = json.loads(message.replace("'", '"'))
                    #print(f"Received tick data: {tick_data}")

                    # Process the tick data here as needed
                    # For example, sending a response back to MT5
                    response_data = {
                        #"response": "Acknowledged",
                        "tick": tick_data
                    }
                    socket_send.send_string(json.dumps(response_data))
                    # print(f"Sent response to MT5: {response_data}")

                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e} for message: {message}")

        except zmq.Again:
            # No message received, just pass
            pass
            # print("No data received yet...")

        # Wait for a short time before the next attempt
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopping Python client...")

# Clean up
socket_send.close()
socket_receive.close()
context.term()
