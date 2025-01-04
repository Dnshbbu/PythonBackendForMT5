import asyncio
import zmq
import zmq.asyncio
import json

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
                        # Parse the JSON-like string
                        tick_data = json.loads(message.replace("'", '"'))
                        # print(f"Received tick data: {tick_data}")

                        # Process the tick data here as needed
                        # For example, sending a response back to MT5
                        response_data = {
                            "tick": tick_data
                        }
                        await socket_send.send_string(json.dumps(response_data))
                        # print(f"Sent response to MT5: {response_data}")

                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e} for message: {message}")

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

