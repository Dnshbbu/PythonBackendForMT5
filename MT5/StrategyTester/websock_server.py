import asyncio
import websockets
import json

async def handler(websocket):
    print(f"Client connected: {websocket.remote_address}")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)  # Parse JSON data
                bid = data.get("bid")
                ask = data.get("ask")
                print(f"Received: Bid={bid}, Ask={ask}")

                # Process the received data as needed

            except json.JSONDecodeError:
                print("Invalid JSON received.")

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Client disconnected: {e.code}, {e.reason}")


async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765): # Replace with your desired port
        print("Server started on ws://0.0.0.0:8765") # Replace with your actual IP and port if needed
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())