import zmq
import time
import json

context = zmq.Context()

# Socket to send commands
socket_send = context.socket(zmq.PUSH)
socket_send.connect("tcp://127.0.0.1:5557")

# Socket to receive responses
socket_receive = context.socket(zmq.PULL)
socket_receive.connect("tcp://127.0.0.1:5556")

print("Python client started. Waiting for MT5 tick data...")

try:
    while True:
        try:
            # Try to receive tick data
            message = socket_receive.recv_string(flags=zmq.NOBLOCK)
            # Parse the JSON-like string
            tick_data = json.loads(message.replace("'", '"'))
            print(f"Received tick data: {tick_data}")
            
            # You can process the tick data here as needed
            
        except zmq.Again:
            pass
            #print("No data received yet...")
        
        # Wait for a short time before the next attempt
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopping Python client...")

# Clean up
socket_send.close()
socket_receive.close()
context.term()