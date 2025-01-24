import asyncio
import zmq
import zmq.asyncio
import json
import logging
from datetime import datetime
import os
import csv
import pandas as pd
import plotly.graph_objects as go

# Fix for Proactor Event Loop Warning on Windows
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class MT5ZMQClient:
    def __init__(self):
        self.context = zmq.asyncio.Context()
        self.socket_receive = self.context.socket(zmq.PULL)
        self.socket_send = self.context.socket(zmq.PUSH)
        self.signal_socket = self.context.socket(zmq.PULL)
        self.is_running = True
        self.headers = {}
        self.message_batch = []
        self.BATCH_SIZE = 40
        
        # Create logs directory if it doesn't exist
        self.logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.setup_logging()

    def setup_logging(self):
        log_file = os.path.join(self.logs_dir, 'mt5_zmq_client.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def get_log_file_path(self, run_id, file_type):
        """Generate full file path for logs"""
        return os.path.join(self.logs_dir, f"{run_id}_{file_type}")

    async def connect(self):
        try:
            self.socket_receive.bind("tcp://127.0.0.1:5556")
            self.socket_send.bind("tcp://127.0.0.1:5557")
            self.signal_socket.bind("tcp://127.0.0.1:5558")
            logging.info("ZMQ sockets bound successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to bind ZMQ sockets: {e}")
            return False

    async def handle_all_details(self, run_id, msg_content):
        """Handle detailed trading information"""
        try:
            csv_filename = self.get_log_file_path(run_id, "all_details.csv")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
            
            with open(csv_filename, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                
                # Handle headers
                if csv_filename not in self.headers:
                    if not os.path.exists(csv_filename) or os.path.getsize(csv_filename) == 0:
                        # Define headers based on the structure we expect from MT5
                        self.headers[csv_filename] = [
                            'Date','Time', 'Symbol', 'Price', 'Equity', 'Balance', 'Profit',
                            'Positions', 'Score', 'ExitScore', 'Factors', 'ExitFactors',
                            'EntryScore', 'ExitScoreDetails', 'Pullback'
                        ]
                        csv_writer.writerow(self.headers[csv_filename])
                        logging.info(f"Created CSV file: {csv_filename} with headers")
                    else:
                        with open(csv_filename, 'r') as f:
                            self.headers[csv_filename] = next(csv.reader(f))
                    return
                
                # Split the message content by the '|' delimiter and clean up the data
                data = {}
                for item in msg_content.split(','):
                    if ':' in item:
                        key, value = item.split(':', 1)
                        data[key.strip()] = value.strip()
                
                # Organize data according to headers
                row_data = []
                for header in self.headers[csv_filename]:
                    if header in data:
                        row_data.append(data[header])
                    else:
                        row_data.append('')  # Add empty string for missing data
                        
                csv_writer.writerow(row_data)
                logging.info(f"Appended new row to {csv_filename}")
                
        except Exception as e:
            logging.error(f"Error handling all details data: {e}")

    async def handle_sequence_of_events(self, run_id, event_content):
        try:
            events_filename = self.get_log_file_path(run_id, "sequence_of_events.log")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            event_line = f"[{timestamp}] {event_content}\n"
            
            with open(events_filename, 'a', encoding='utf-8') as f:
                f.write(event_line)
            
            logging.info(f"Event logged to {events_filename}: {event_content}")
            
        except Exception as e:
            logging.error(f"Error handling sequence of events: {e}")

    async def handle_csv_data(self, run_id, msg_type, msg_content):
        try:
            csv_filename = self.get_log_file_path(run_id, f"{msg_type}.csv")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
            
            with open(csv_filename, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)

                if msg_type == 'priceandequity':
                    if isinstance(msg_content, list):
                        for row in msg_content:
                            csv_writer.writerow(row.split(','))
                        logging.info(f"Appended {len(msg_content)} rows to {csv_filename}")
                
                elif msg_type == 'transactions':
                    if csv_filename not in self.headers:
                        if not os.path.exists(csv_filename) or os.path.getsize(csv_filename) == 0:
                            self.headers[csv_filename] = msg_content.split(',')
                            csv_writer.writerow(self.headers[csv_filename])
                            logging.info(f"Created CSV file: {csv_filename} with headers")
                        else:
                            with open(csv_filename, 'r') as f:
                                self.headers[csv_filename] = next(csv.reader(f))
                        return

                    data_row = msg_content.split(',')
                    if len(data_row) == len(self.headers[csv_filename]):
                        csv_writer.writerow(data_row)
                    else:
                        logging.error(f"Invalid data row format in {csv_filename}")

        except Exception as e:
            logging.error(f"Error handling CSV data: {e}")

    def draw_plot(self, run_id):
        try:
            equity_price_file = self.get_log_file_path(run_id, "priceandequity.csv")
            transactions_file = self.get_log_file_path(run_id, "transactions.csv")
            
            df_equity_price = pd.read_csv(equity_price_file)
            df_equity_price['DateTime'] = pd.to_datetime(df_equity_price['Date'] + ' ' + df_equity_price['Time'])

            df_transactions = pd.read_csv(transactions_file)
            df_transactions['DateTime'] = pd.to_datetime(df_transactions['Date'] + ' ' + df_transactions['Time'])

            # Create figure
            fig = go.Figure()

            # Add price trace
            fig.add_trace(go.Scatter(
                x=df_equity_price['DateTime'],
                y=df_equity_price['Price'],
                mode='lines',
                name='Price',
                yaxis='y1'
            ))

            # Add equity trace
            fig.add_trace(go.Scatter(
                x=df_equity_price['DateTime'],
                y=df_equity_price['Equity'],
                mode='lines',
                name='Equity',
                yaxis='y2'
            ))

            # Add buy/sell signals
            df_buy = df_transactions[df_transactions['Type'].str.contains('Buy')]
            df_sell = df_transactions[df_transactions['Type'].str.contains('Sell')]

            # Add buy signals
            fig.add_trace(go.Scatter(
                x=df_buy['DateTime'],
                y=df_equity_price.loc[df_equity_price['DateTime'].isin(df_buy['DateTime']), 'Price'],
                mode='markers',
                marker=dict(symbol='triangle-up', color='green', size=10),
                name='Buy'
            ))

            # Add sell signals
            fig.add_trace(go.Scatter(
                x=df_sell['DateTime'],
                y=df_equity_price.loc[df_equity_price['DateTime'].isin(df_sell['DateTime']), 'Price'],
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=10),
                name='Sell'
            ))

            # Update layout
            fig.update_layout(
                title='Equity and Price with Trading Signals',
                xaxis_title='DateTime',
                yaxis_title='Price',
                yaxis2=dict(title='Equity', overlaying='y', side='right'),
                xaxis_rangeslider_visible=True
            )

            # Save plot
            output_file = self.get_log_file_path(run_id, "output_plot.html")
            fig.write_html(output_file)
            logging.info(f"Plot saved as {output_file}")

        except Exception as e:
            logging.error(f"Error creating plot: {e}")

    async def process_message(self, message):
        try:
            logging.debug(f"Raw message received: {message}")
            
            try:
                data = json.loads(message)
                logging.info(f"Parsed message from MT5: {data}")
            except json.JSONDecodeError:
                logging.warning(f"Could not parse as JSON: {message}")
                return False

            run_id = data.get("run_id")
            msg_type = data.get("type", "").strip().lower()
            msg_content = data.get("msg")

            if run_id and msg_type and msg_content:
                if msg_type == 'draw_plot':
                    self.draw_plot(run_id)
                elif msg_type == 'sequenceofevents':
                    await self.handle_sequence_of_events(run_id, msg_content)
                elif msg_type == 'alldetails':
                    await self.handle_all_details(run_id, msg_content)
                else:
                    await self.handle_csv_data(run_id, msg_type, msg_content)
            
            return True

        except Exception as e:
            logging.error(f"Error processing message: {str(e)}\nMessage: {message}")
            return False

    async def send_to_mt5(self, message):
        try:
            if not isinstance(message, str):
                message = json.dumps(message)
            
            logging.debug(f"Sending to MT5: {message}")
            await self.socket_send.send_string(message)
            logging.info("Successfully sent message to MT5")
            return True
        except Exception as e:
            logging.error(f"Error sending to MT5: {str(e)}\nMessage: {message}")
            return False

    async def send_signal(self, signal_value):
        message = {
            "type": "new_signal",
            "timestamp": datetime.now().isoformat(),
            "value": signal_value,
            "run_id": "python_signal"
        }
        await self.send_to_mt5(json.dumps(message))
        logging.info(f"Sent signal to MT5: {signal_value}")

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
                except asyncio.TimeoutError:
                    pass

                await asyncio.sleep(0.01)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)

    async def cleanup(self):
        logging.info("Cleaning up resources...")
        self.socket_send.close()
        self.socket_receive.close()
        self.signal_socket.close()
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