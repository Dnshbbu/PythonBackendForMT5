# import asyncio
# import zmq
# import zmq.asyncio
# import csv
# import json
# import os
# from datetime import datetime
# import pandas as pd
# import plotly.graph_objects as go
# import logging

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('mt5_zmq_client.log'),
#         logging.StreamHandler()
#     ]
# )

# # Fix for Proactor Event Loop Warning on Windows
# if os.name == 'nt':
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# class MT5ZMQClient:
#     def __init__(self):
#         self.context = zmq.asyncio.Context()
#         self.socket_send = self.context.socket(zmq.PUSH)
#         self.socket_receive = self.context.socket(zmq.PULL)
#         self.headers = {}
#         self.is_running = True

#     async def connect(self):
#         """Connect to MT5 ZMQ sockets"""
#         try:
#             self.socket_send.connect("tcp://127.0.0.1:5557")
#             self.socket_receive.connect("tcp://127.0.0.1:5556")
#             logging.info("Connected to MT5 ZMQ sockets")
#             return True
#         except Exception as e:
#             logging.error(f"Failed to connect to MT5: {e}")
#             return False

#     async def send_command(self, command):
#         """Send a command to MT5"""
#         try:
#             cmd_str = f"CMD:{command}"
#             await self.socket_send.send_string(cmd_str)
#             logging.info(f"Sent command to MT5: {cmd_str}")
#             return True
#         except Exception as e:
#             logging.error(f"Error sending command to MT5: {e}")
#             return False

#     def process_price_equity_data(self, msg_content, csv_filename):
#         """Process price and equity data"""
#         try:
#             with open(csv_filename, 'a', newline='') as csvfile:
#                 csv_writer = csv.writer(csvfile)
#                 if isinstance(msg_content, list):
#                     for row in msg_content:
#                         csv_writer.writerow(row.split(','))
#                     logging.debug(f"Appended {len(msg_content)} rows to {csv_filename}")
#         except Exception as e:
#             logging.error(f"Error processing price/equity data: {e}")

#     def process_transaction_data(self, msg_content, csv_filename):
#         """Process transaction data"""
#         try:
#             with open(csv_filename, 'a', newline='') as csvfile:
#                 csv_writer = csv.writer(csvfile)
                
#                 # Handle headers
#                 if csv_filename not in self.headers:
#                     if os.path.getsize(csv_filename) == 0:
#                         self.headers[csv_filename] = msg_content.split(',')
#                         csv_writer.writerow(self.headers[csv_filename])
#                         logging.info(f"Created headers in {csv_filename}")
#                         return
#                     else:
#                         with open(csv_filename, 'r') as f:
#                             self.headers[csv_filename] = next(csv.reader(f))
                
#                 # Process data row
#                 data_row = msg_content.split(',')
#                 if len(data_row) == len(self.headers[csv_filename]):
#                     csv_writer.writerow(data_row)
#                     logging.debug(f"Appended transaction data to {csv_filename}")
#                 else:
#                     logging.error(f"Data row length mismatch. Expected: {len(self.headers[csv_filename])}, Got: {len(data_row)}")
#         except Exception as e:
#             logging.error(f"Error processing transaction data: {e}")

#     def draw_plot(self, run_id):
#         """Create interactive plot using plotly"""
#         try:
#             # Load data files
#             equity_price_file = f"{run_id}_priceandequity.csv"
#             transactions_file = f"{run_id}_transactions.csv"
            
#             # Read and process equity/price data
#             df_equity_price = pd.read_csv(equity_price_file)
#             df_equity_price['DateTime'] = pd.to_datetime(df_equity_price['Date'] + ' ' + df_equity_price['Time'])
            
#             # Read and process transactions data
#             df_transactions = pd.read_csv(transactions_file)
#             df_transactions['DateTime'] = pd.to_datetime(df_transactions['Time'])
            
#             # Create plot
#             fig = go.Figure()
            
#             # Add price trace
#             fig.add_trace(go.Scatter(
#                 x=df_equity_price['DateTime'],
#                 y=df_equity_price['Price'],
#                 mode='lines',
#                 name='Price',
#                 yaxis='y1',
#                 hovertemplate='<b>Price</b><br>Date: %{x}<br>Price: %{y}<extra></extra>'
#             ))
            
#             # Add equity trace
#             fig.add_trace(go.Scatter(
#                 x=df_equity_price['DateTime'],
#                 y=df_equity_price['Equity'],
#                 mode='lines',
#                 name='Equity',
#                 yaxis='y2',
#                 hovertemplate='<b>Equity</b><br>Date: %{x}<br>Equity: %{y}<extra></extra>'
#             ))
            
#             # Add trade signals
#             df_long = df_transactions[df_transactions['Action'].str.contains('Buy', na=False)]
            
#             if not df_long.empty:
#                 fig.add_trace(go.Scatter(
#                     x=df_long['DateTime'],
#                     y=df_equity_price.loc[df_equity_price['DateTime'].isin(df_long['DateTime']), 'Price'],
#                     mode='markers',
#                     marker=dict(symbol='triangle-up', color='green', size=10),
#                     name='Buy',
#                     hovertemplate='<b>Buy</b><br>Date: %{x}<br>Price: %{y}<br>Score: %{customdata[0]}<extra></extra>',
#                     customdata=df_long[['Total Score']].values
#                 ))
            
#             # Update layout
#             fig.update_layout(
#                 title=f'Trading Activity - {run_id}',
#                 xaxis_title='DateTime',
#                 yaxis_title='Price',
#                 yaxis2=dict(title='Equity', overlaying='y', side='right'),
#                 xaxis_rangeslider_visible=True,
#                 hovermode='x unified'
#             )
            
#             # Save plot
#             output_file = f"{run_id}_plot.html"
#             fig.write_html(output_file)
#             logging.info(f"Plot saved as {output_file}")
            
#         except Exception as e:
#             logging.error(f"Error creating plot: {e}")

#     async def process_message(self, message):
#         """Process incoming message from MT5"""
#         try:
#             data = json.loads(message)
#             run_id = data.get("run_id")
#             msg_type = data.get("type", "").strip().lower()
#             msg_content = data.get("msg")
            
#             if run_id and msg_type and msg_content:
#                 # Generate CSV filename
#                 csv_filename = f"{run_id}_{msg_type}.csv"
                
#                 if msg_type == 'draw_plot':
#                     self.draw_plot(run_id)
#                 elif msg_type == 'priceandequity':
#                     self.process_price_equity_data(msg_content, csv_filename)
#                 elif msg_type == 'transactions':
#                     self.process_transaction_data(msg_content, csv_filename)
                
#                 # Send acknowledgment
#                 await self.send_command("RECEIVED")
                
#             return True
#         except json.JSONDecodeError as e:
#             logging.error(f"JSON decode error: {e}")
#             return False
#         except Exception as e:
#             logging.error(f"Error processing message: {e}")
#             return False

#     async def run(self):
#         """Main loop for the ZMQ client"""
#         if not await self.connect():
#             return
        
#         logging.info("Starting MT5 ZMQ client...")
        
#         while self.is_running:
#             try:
#                 # Try to receive data with timeout
#                 try:
#                     message = await asyncio.wait_for(
#                         self.socket_receive.recv_string(),
#                         timeout=0.1
#                     )
#                     if message:
#                         await self.process_message(message)
#                 except asyncio.TimeoutError:
#                     await asyncio.sleep(0.01)
#                     continue
                
#             except asyncio.CancelledError:
#                 logging.info("Received shutdown signal")
#                 break
#             except Exception as e:
#                 logging.error(f"Error in main loop: {e}")
#                 await asyncio.sleep(1)  # Delay before retry
                
#         await self.cleanup()

#     async def cleanup(self):
#         """Cleanup resources"""
#         logging.info("Cleaning up resources...")
#         self.socket_send.close()
#         self.socket_receive.close()
#         self.context.term()

# async def main():
#     client = MT5ZMQClient()
#     try:
#         await client.run()
#     except KeyboardInterrupt:
#         logging.info("Keyboard interrupt received")
#         client.is_running = False
#     except Exception as e:
#         logging.error(f"Unexpected error: {e}")
#     finally:
#         await client.cleanup()

# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         logging.info("Application terminated by user")
























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


# class MT5ZMQClient:
#     def __init__(self):
#         self.context = zmq.asyncio.Context()
#         # PULL socket for receiving FROM MT5
#         self.socket_receive = self.context.socket(zmq.PULL)
#         # PUSH socket for sending TO MT5
#         self.socket_send = self.context.socket(zmq.PUSH)
#         self.is_running = True
#         self.setup_logging()

#     def setup_logging(self):
#         logging.basicConfig(
#             level=logging.INFO,
#             format='%(asctime)s - %(levelname)s - %(message)s',
#             handlers=[
#                 logging.FileHandler('mt5_zmq_client.log'),
#                 logging.StreamHandler()
#             ]
#         )

#     async def connect(self):
#         """Connect to MT5 ZMQ sockets"""
#         try:
#             # Bind sockets (MT5 will connect to these)
#             self.socket_receive.bind("tcp://127.0.0.1:5556")  # For receiving FROM MT5
#             self.socket_send.bind("tcp://127.0.0.1:5557")     # For sending TO MT5
#             logging.info("ZMQ sockets bound successfully")
#             return True
#         except Exception as e:
#             logging.error(f"Failed to bind ZMQ sockets: {e}")
#             return False

#     async def send_to_mt5(self, message):
#         """Send a message to MT5"""
#         try:
#             await self.socket_send.send_string(message)
#             logging.info(f"Sent to MT5: {message}")
#             return True
#         except Exception as e:
#             logging.error(f"Error sending to MT5: {e}")
#             return False

#     async def process_message(self, message):
#         """Process received message from MT5"""
#         try:
#             data = json.loads(message)
#             logging.info(f"Received from MT5: {data}")
            
#             # Send acknowledgment back to MT5
#             response = {"status": "received", "timestamp": datetime.now().isoformat()}
#             await self.send_to_mt5(json.dumps(response))
            
#             return True
#         except json.JSONDecodeError as e:
#             logging.error(f"JSON decode error: {e}")
#             return False
#         except Exception as e:
#             logging.error(f"Error processing message: {e}")
#             return False

#     async def run(self):
#         """Main loop for the ZMQ client"""
#         if not await self.connect():
#             return

#         logging.info("Starting MT5 ZMQ client...")
        
#         while self.is_running:
#             try:
#                 # Try to receive data with timeout
#                 try:
#                     message = await asyncio.wait_for(
#                         self.socket_receive.recv_string(),
#                         timeout=0.1
#                     )
#                     if message:
#                         await self.process_message(message)
#                 except asyncio.TimeoutError:
#                     await asyncio.sleep(0.01)
#                     continue
                
#                 # Periodically send a heartbeat to MT5
#                 await self.send_to_mt5(json.dumps({
#                     "type": "heartbeat",
#                     "timestamp": datetime.now().isoformat()
#                 }))
                
#             except asyncio.CancelledError:
#                 logging.info("Received shutdown signal")
#                 break
#             except Exception as e:
#                 logging.error(f"Error in main loop: {e}")
#                 await asyncio.sleep(1)
                
#         await self.cleanup()

#     async def cleanup(self):
#         """Cleanup resources"""
#         logging.info("Cleaning up resources...")
#         self.socket_send.close()
#         self.socket_receive.close()
#         self.context.term()

# async def main():
#     client = MT5ZMQClient()
#     try:
#         await client.run()
#     except KeyboardInterrupt:
#         logging.info("Keyboard interrupt received")
#         client.is_running = False
#     finally:
#         await client.cleanup()

# if __name__ == "__main__":
#     asyncio.run(main())


class MT5ZMQClient:
    def __init__(self):
        self.context = zmq.asyncio.Context()
        # PULL socket for receiving FROM MT5
        self.socket_receive = self.context.socket(zmq.PULL)
        # PUSH socket for sending TO MT5
        self.socket_send = self.context.socket(zmq.PUSH)
        self.is_running = True
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('mt5_zmq_client.log'),
                logging.StreamHandler()
            ]
        )

    async def connect(self):
        """Connect to MT5 ZMQ sockets"""
        try:
            # Bind sockets (MT5 will connect to these)
            self.socket_receive.bind("tcp://127.0.0.1:5556")  # For receiving FROM MT5
            self.socket_send.bind("tcp://127.0.0.1:5557")     # For sending TO MT5
            logging.info("ZMQ sockets bound successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to bind ZMQ sockets: {e}")
            return False

    async def send_to_mt5(self, message):
        """Send a message to MT5"""
        try:
            await self.socket_send.send_string(message)
            logging.info(f"Sent to MT5: {message}")
            return True
        except Exception as e:
            logging.error(f"Error sending to MT5: {e}")
            return False

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
            
            # Send detailed acknowledgment back to MT5
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

    async def run(self):
        """Main loop for the ZMQ client"""
        if not await self.connect():
            return

        logging.info("Starting MT5 ZMQ client...")
        
        while self.is_running:
            try:
                # Try to receive data with timeout
                try:
                    message = await asyncio.wait_for(
                        self.socket_receive.recv_string(),
                        timeout=0.1
                    )
                    if message:
                        await self.process_message(message)
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.01)
                    continue
                
                # Periodically send a heartbeat to MT5
                await self.send_to_mt5(json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                }))
                
            except asyncio.CancelledError:
                logging.info("Received shutdown signal")
                break
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)
                
        await self.cleanup()

    async def cleanup(self):
        """Cleanup resources"""
        logging.info("Cleaning up resources...")
        self.socket_send.close()
        self.socket_receive.close()
        self.context.term()

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