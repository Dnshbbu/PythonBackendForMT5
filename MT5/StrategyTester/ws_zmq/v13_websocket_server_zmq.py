
import asyncio
import zmq
import zmq.asyncio
import json
import logging
from datetime import datetime
import os

# Fix for Proactor Event Loop Warning on Windows
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class MT5ZMQClient:
    def __init__(self):
        self.context = zmq.asyncio.Context()
        # PULL socket for receiving FROM MT5
        self.socket_receive = self.context.socket(zmq.PULL)
        # PUSH socket for sending TO MT5
        self.socket_send = self.context.socket(zmq.PUSH)
        # New PULL socket for receiving signals from send_signal.py
        self.signal_socket = self.context.socket(zmq.PULL)
        self.is_running = True
        self.setup_logging()

    async def connect(self):
        """Connect to MT5 ZMQ sockets"""
        try:
            # Bind sockets (MT5 will connect to these)
            self.socket_receive.bind("tcp://127.0.0.1:5556")  # For receiving FROM MT5
            self.socket_send.bind("tcp://127.0.0.1:5557")     # For sending TO MT5
            self.signal_socket.bind("tcp://127.0.0.1:5558")   # For receiving signals
            logging.info("ZMQ sockets bound successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to bind ZMQ sockets: {e}")
            return False
    # def __init__(self):
    #     self.context = zmq.asyncio.Context()
    #     # PULL socket for receiving FROM MT5
    #     self.socket_receive = self.context.socket(zmq.PULL)
    #     # PUSH socket for sending TO MT5
    #     self.socket_send = self.context.socket(zmq.PUSH)
    #     self.is_running = True
    #     self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('mt5_zmq_client.log'),
                logging.StreamHandler()
            ]
        )

    # async def connect(self):
    #     """Connect to MT5 ZMQ sockets"""
    #     try:
    #         # Bind sockets (MT5 will connect to these)
    #         self.socket_receive.bind("tcp://127.0.0.1:5556")  # For receiving FROM MT5
    #         self.socket_send.bind("tcp://127.0.0.1:5557")     # For sending TO MT5
    #         logging.info("ZMQ sockets bound successfully")
    #         return True
    #     except Exception as e:
    #         logging.error(f"Failed to bind ZMQ sockets: {e}")
    #         return False


    async def send_to_mt5(self, message):
        """Send a message to MT5"""
        try:
            await self.socket_send.send_string(message)
            logging.info(f"Sent to MT5: {message}")
            return True
        except Exception as e:
            logging.error(f"Error sending to MT5: {e}")
            return False

    # async def process_message(self, message):
    #     """Process received message from MT5"""
    #     try:
    #         # Log raw message first
    #         logging.debug(f"Raw message received: {message}")
            
    #         try:
    #             data = json.loads(message)
    #             logging.info(f"Parsed message from MT5: {data}")
    #         except json.JSONDecodeError:
    #             logging.warning(f"Could not parse as JSON: {message}")
    #             data = {"raw_message": message}
            
    #         # Check if this is a confirmation message
    #         if isinstance(data, dict) and "status" in data and data["status"] == "mt5_received":
    #             logging.info("MT5 confirmed receipt of previous message")
    #             return True
            
    #         # Send detailed acknowledgment back to MT5
    #         response = {
    #             "status": "received",
    #             "timestamp": datetime.now().isoformat(),
    #             "message_type": data.get("type", "unknown"),
    #             "run_id": data.get("run_id", "unknown")
    #         }
            
    #         response_json = json.dumps(response)
    #         logging.debug(f"Sending response to MT5: {response_json}")
    #         await self.send_to_mt5(response_json)
            
    #         return True
            
    #     except Exception as e:
    #         logging.error(f"Error processing message: {str(e)}\nMessage: {message}")
    #         return False


    async def process_message(self, message):
        """Process received message from MT5"""
        try:
            # Log raw message first
            logging.debug(f"Raw message received: {message}")
            
            try:
                data = json.loads(message)
                logging.info(f"Parsed message from MT5: {data}")
            except json.JSONDecodeError:
                logging.warning(f"Could not parse as JSON: {message}")
                data = {"raw_message": message}
            
            # Check if this is a confirmation message
            if isinstance(data, dict) and "status" in data and data["status"] == "mt5_received":
                logging.info("MT5 confirmed receipt of previous message")
                return True
            
            # Don't send acknowledgment for every message
            # Only send acknowledgment for specific message types that need it
            if data.get("type") != "new_signal":
                response = {
                    "status": "received",
                    "timestamp": datetime.now().isoformat(),
                    "message_type": data.get("type", "unknown"),
                    "run_id": data.get("run_id", "unknown")
                }
                
                response_json = json.dumps(response)
                logging.debug(f"Sending response to MT5: {response_json}")
                await self.send_to_mt5(response_json)
            
            return True
            
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}\nMessage: {message}")
            return False


    async def send_to_mt5(self, message):
        """Send a message to MT5 with error handling"""
        try:
            # Ensure message is properly JSON-encoded
            if not isinstance(message, str):
                message = json.dumps(message)
            
            logging.debug(f"Sending to MT5: {message}")
            await self.socket_send.send_string(message)
            logging.info("Successfully sent message to MT5")
            return True
        except Exception as e:
            logging.error(f"Error sending to MT5: {str(e)}\nMessage: {message}")
            return False

    # async def run(self):
    #     """Main loop for the ZMQ client"""
    #     if not await self.connect():
    #         return

    #     logging.info("Starting MT5 ZMQ client...")
        
    #     while self.is_running:
    #         try:
    #             # Try to receive data with timeout
    #             try:
    #                 message = await asyncio.wait_for(
    #                     self.socket_receive.recv_string(),
    #                     timeout=0.1
    #                 )
    #                 if message:
    #                     await self.process_message(message)
    #             except asyncio.TimeoutError:
    #                 await asyncio.sleep(0.01)
    #                 continue
                
    #             # Periodically send a heartbeat to MT5
    #             await self.send_to_mt5(json.dumps({
    #                 "type": "heartbeat",
    #                 "timestamp": datetime.now().isoformat()
    #             }))
                
    #         except asyncio.CancelledError:
    #             logging.info("Received shutdown signal")
    #             break
    #         except Exception as e:
    #             logging.error(f"Error in main loop: {e}")
    #             await asyncio.sleep(1)
                
    #     await self.cleanup()


    async def run(self):
        if not await self.connect():
            return

        logging.info("Starting MT5 ZMQ client...")
        
        while self.is_running:
            try:
                # Check for MT5 messages
                try:
                    message = await asyncio.wait_for(
                        self.socket_receive.recv_string(),
                        timeout=0.1
                    )
                    if message:
                        await self.process_message(message)
                except asyncio.TimeoutError:
                    pass

                # Check for signal messages
                try:
                    signal = await asyncio.wait_for(
                        self.signal_socket.recv_string(),
                        timeout=0.1
                    )
                    if signal:
                        logging.info(f"Received signal to forward: {signal}")
                        await self.socket_send.send_string(signal)
                        logging.info("Signal forwarded to MT5")
                except asyncio.TimeoutError:
                    pass

                await asyncio.sleep(0.01)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)


    async def cleanup(self):
        """Cleanup resources"""
        logging.info("Cleaning up resources...")
        self.socket_send.close()
        self.socket_receive.close()
        self.context.term()



    # # Add to v13_websocket_server_zmq.py
    # async def send_signal(self, signal_value):
    #     """Send a signal to MT5"""
    #     message = {
    #         "type": "new_signal",
    #         "timestamp": datetime.now().isoformat(),
    #         "value": signal_value,
    #         "run_id": "python_signal"
    #     }
        
    #     await self.send_to_mt5(json.dumps(message))
    #     logging.info(f"Sent signal to MT5: {signal_value}")


    # Add to v13_websocket_server_zmq.py
    async def send_signal(self, signal_value):
        """Send a signal to MT5"""
        message = {
            "type": "new_signal",
            "timestamp": datetime.now().isoformat(),
            "value": signal_value,
            "run_id": "python_signal"  # Match your run_id format
        }
        
        await self.send_to_mt5(json.dumps(message))
        logging.info(f"Sent signal to MT5: {signal_value}")

# # You can then call it like this:
# await client.send_signal(1.5)  # Example signal value


async def main():
    client = MT5ZMQClient()
    try:
        await client.run()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received")
        client.is_running = False
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())