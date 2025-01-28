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
from database_manager import DatabaseManager
from real_time_price_predictor import RealTimePricePredictor
from model_training_manager import ModelTrainingManager
from prediction_tracker import PredictionTracker
from config import ZMQ_CONFIG, DATABASE_CONFIG, MODEL_CONFIG

# Fix for Proactor Event Loop Warning on Windows
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class MT5ZMQClient:

    # def __init__(self):
    #     self.context = zmq.asyncio.Context()
    #     self.socket_receive = self.context.socket(zmq.PULL)
    #     self.socket_send = self.context.socket(zmq.PUSH)
    #     self.signal_socket = self.context.socket(zmq.PULL)
    #     self.is_running = True
    #     self.headers = {}
    #     self.message_batch = []
    #     # self.BATCH_SIZE = 40
    #     self.batch_size = MODEL_CONFIG['batch_size']
        
    #     # Create logs directory if it doesn't exist
    #     self.logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    #     os.makedirs(self.logs_dir, exist_ok=True)
        
    #     # Initialize model directories
    #     self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    #     os.makedirs(self.models_dir, exist_ok=True)
        
    #     self.setup_logging()
        
    #     # Initialize training manager
    #     self.training_manager = ModelTrainingManager(
    #         db_path=os.path.join(self.logs_dir, 'trading_data.db'),
    #         models_dir=self.models_dir,
    #         min_rows_for_training=20  # Trigger retraining every 20 new rows
    #     )
        
    #     # Clean up any invalid metrics data
    #     self.training_manager.cleanup_invalid_metrics()

    def __init__(self):
        """
        Initialize the MT5ZMQClient with proper configuration and component setup.
        """
        try:
            # Initialize ZMQ context and sockets
            self.context = zmq.asyncio.Context()
            self.socket_receive = self.context.socket(zmq.PULL)
            self.socket_send = self.context.socket(zmq.PUSH)
            self.signal_socket = self.context.socket(zmq.PULL)
            self.is_running = True

            # Initialize data structures
            self.headers = {}
            self.message_batch = []
            self.batch_size = MODEL_CONFIG['batch_size']
            
            # Create logs directory if it doesn't exist
            self.logs_dir = DATABASE_CONFIG['logs_dir']
            self.models_dir = DATABASE_CONFIG['models_dir']
            os.makedirs(self.logs_dir, exist_ok=True)
            os.makedirs(self.models_dir, exist_ok=True)
            
            # Setup logging first so we can track initialization
            self.setup_logging()
            logging.info("Initializing MT5 ZMQ Client...")

            # Initialize database manager
            try:
                self.db_manager = DatabaseManager()
                logging.info("Database manager initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize database manager: {e}")
                raise

            # Initialize training manager with configuration
            try:
                self.training_manager = ModelTrainingManager(
                    db_path=os.path.join(self.logs_dir, DATABASE_CONFIG['db_name']),
                    models_dir=self.models_dir,
                    min_rows_for_training=MODEL_CONFIG['min_rows_for_training']
                )
                logging.info("Training manager initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize training manager: {e}")
                raise

            # Initialize price predictor with configuration
            try:
                self.price_predictor = RealTimePricePredictor(
                    db_path=os.path.join(self.logs_dir, DATABASE_CONFIG['db_name']),
                    models_dir=self.models_dir,
                    batch_size=MODEL_CONFIG['batch_size'],
                    training_manager=self.training_manager
                )
                logging.info("Price predictor initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize price predictor: {e}")
                raise

            # Initialize prediction tracker
            try:
                self.prediction_tracker = PredictionTracker(
                    db_path=os.path.join(self.logs_dir, DATABASE_CONFIG['db_name'])
                )
                logging.info("Prediction tracker initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize prediction tracker: {e}")
                raise

            # Clean up any invalid metrics data from previous runs
            try:
                self.training_manager.cleanup_invalid_metrics()
                logging.info("Cleaned up invalid metrics data")
            except Exception as e:
                logging.warning(f"Non-critical error during metrics cleanup: {e}")

            logging.info("MT5 ZMQ Client initialization completed successfully")

        except Exception as e:
            logging.error(f"Critical error during MT5 ZMQ Client initialization: {e}")
            # Clean up any partially initialized resources
            # self.cleanup_resources()
            raise

    async def handle_all_details(self, run_id, msg_content):
        """Handle detailed trading information with ML integration and real-time predictions"""
        try:
            csv_filename = self.get_log_file_path(run_id, "all_details.csv")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
            
            # Initialize components if not exists
            if not hasattr(self, 'price_predictor'):
                db_path = os.path.join(self.logs_dir, 'trading_data.db')
                self.price_predictor = RealTimePricePredictor(
                    db_path=db_path, 
                    models_dir=self.models_dir, 
                    batch_size=10,
                    training_manager=self.training_manager
                )
                    
            if not hasattr(self, 'db_manager'):
                db_path = os.path.join(self.logs_dir, 'trading_data.db')
                self.db_manager = DatabaseManager(db_path)
                
            if not hasattr(self, 'prediction_tracker'):
                db_path = os.path.join(self.logs_dir, 'trading_data.db')
                self.prediction_tracker = PredictionTracker(db_path)
                
            # Process the message content into a dictionary
            data = {}
            for item in msg_content.split(','):
                if ':' in item:
                    key, value = item.split(':', 1)
                    data[key.strip()] = value.strip()
            
            # Handle CSV operations
            with open(csv_filename, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                
                # Handle headers
                if csv_filename not in self.headers:
                    if not os.path.exists(csv_filename) or os.path.getsize(csv_filename) == 0:
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
                
                # Write row data
                row_data = []
                for header in self.headers[csv_filename]:
                    if header in data:
                        row_data.append(data[header])
                    else:
                        row_data.append('')
                csv_writer.writerow(row_data)
            
            # Handle DB operations
            try:
                table_name = self.db_manager.create_table_for_strategy(run_id, data)
                self.db_manager.insert_data(table_name, data)
                
                # Check if retraining is needed and trigger if necessary
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.training_manager.check_and_trigger_training,
                    table_name
                )
                
            except Exception as db_error:
                logging.error(f"Database operation failed: {db_error}")
                logging.exception(db_error)
            
            # Make real-time prediction
            try:
                prediction_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.price_predictor.add_data_point,
                    data
                )
                
                if prediction_result is not None:
                    # Get latest training status
                    training_status = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.training_manager.get_latest_training_status
                    )

                    # Get current model name
                    current_model = getattr(self.price_predictor.model_predictor, 'current_model_name', 'unknown')
                    
                    # Create prediction message with training status
                    prediction_message = {
                        "type": "price_prediction",
                        "run_id": run_id,
                        "timestamp": datetime.now().isoformat(),
                        "prediction": float(prediction_result['prediction']),
                        "confidence": float(prediction_result['confidence']),
                        "is_confident": bool(prediction_result['is_confident']),
                        "current_price": float(data.get('Price', 0)),
                        "top_features": prediction_result['top_features'],
                        "model_info": {
                            "current_model": current_model,
                            "is_training": self.training_manager.is_training,
                            "last_training": training_status
                        }
                    }
                    
                    # Record prediction vs actual
                    actual_price = float(data.get('Price', 0))
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.prediction_tracker.record_prediction,
                        run_id,
                        actual_price,
                        prediction_message
                    )
                    
                    # Get latest metrics
                    metrics = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.prediction_tracker.get_latest_metrics,
                        run_id
                    )
                    
                    # Add metrics to prediction message
                    prediction_message["performance_metrics"] = metrics
                    
                    # Send prediction to MT5
                    await self.send_to_mt5(json.dumps(prediction_message))
                    logging.info(f"Sent prediction to MT5 using model {current_model}: "
                            f"{prediction_result['prediction']:.4f} "
                            f"(confidence: {prediction_result['confidence']:.4f})")
            
            except Exception as pred_error:
                logging.error(f"Prediction failed: {pred_error}")
                logging.exception(pred_error)
            
        except Exception as e:
            logging.error(f"Error handling all details data: {e}")
            logging.exception(e)




    async def cleanup(self):
        logging.info("Cleaning up resources...")
        self.socket_send.close()
        self.socket_receive.close()
        self.signal_socket.close()
        self.context.term()
        
        # Additional cleanup for training manager if needed
        if hasattr(self, 'training_manager'):
            # Wait for any ongoing training to complete
            while self.training_manager.is_training:
                await asyncio.sleep(1)


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

    # async def connect(self):
    #     try:
    #         self.socket_receive.bind("tcp://127.0.0.1:5556")
    #         self.socket_send.bind("tcp://127.0.0.1:5557")
    #         self.signal_socket.bind("tcp://127.0.0.1:5558")
    #         logging.info("ZMQ sockets bound successfully")
    #         return True
    #     except Exception as e:
    #         logging.error(f"Failed to bind ZMQ sockets: {e}")
    #         return False
        

    async def connect(self):
        try:
            self.socket_receive.bind(ZMQ_CONFIG['receive_address'])
            self.socket_send.bind(ZMQ_CONFIG['send_address'])
            self.signal_socket.bind(ZMQ_CONFIG['signal_address'])
            logging.info("ZMQ sockets bound successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to bind ZMQ sockets: {e}")
            return False











 

    # async def handle_all_details(self, run_id, msg_content):
    #     """Handle detailed trading information with ML integration and real-time predictions"""
    #     try:
    #         csv_filename = self.get_log_file_path(run_id, "all_details.csv")
            
    #         # Create directory if it doesn't exist
    #         os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
            
    #         # Initialize real-time predictor if not exists
    #         if not hasattr(self, 'price_predictor'):
    #             db_path = os.path.join(self.logs_dir, 'trading_data.db')
    #             models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    #             self.price_predictor = RealTimePricePredictor(db_path, models_dir, batch_size=10)
                    
    #         # Initialize DB manager if not exists
    #         if not hasattr(self, 'db_manager'):
    #             db_path = os.path.join(self.logs_dir, 'trading_data.db')
    #             self.db_manager = DatabaseManager(db_path)
                
    #         # Process the message content into a dictionary
    #         data = {}
    #         for item in msg_content.split(','):
    #             if ':' in item:
    #                 key, value = item.split(':', 1)
    #                 data[key.strip()] = value.strip()
            
    #         # Handle CSV operations
    #         with open(csv_filename, 'a', newline='') as csvfile:
    #             csv_writer = csv.writer(csvfile)
                
    #             # Handle headers
    #             if csv_filename not in self.headers:
    #                 if not os.path.exists(csv_filename) or os.path.getsize(csv_filename) == 0:
    #                     self.headers[csv_filename] = [
    #                         'Date','Time', 'Symbol', 'Price', 'Equity', 'Balance', 'Profit',
    #                         'Positions', 'Score', 'ExitScore', 'Factors', 'ExitFactors',
    #                         'EntryScore', 'ExitScoreDetails', 'Pullback'
    #                     ]
    #                     csv_writer.writerow(self.headers[csv_filename])
    #                     logging.info(f"Created CSV file: {csv_filename} with headers")
    #                 else:
    #                     with open(csv_filename, 'r') as f:
    #                         self.headers[csv_filename] = next(csv.reader(f))
    #                 return
                
    #             # Write row data
    #             row_data = []
    #             for header in self.headers[csv_filename]:
    #                 if header in data:
    #                     row_data.append(data[header])
    #                 else:
    #                     row_data.append('')
    #             csv_writer.writerow(row_data)
            
    #         # Handle DB operations
    #         try:
    #             table_name = self.db_manager.create_table_for_strategy(run_id, data)
    #             self.db_manager.insert_data(table_name, data)
    #         except Exception as db_error:
    #             logging.error(f"Database operation failed: {db_error}")
    #             logging.exception(db_error)
            
    #         # Make real-time prediction
    #         try:
    #             prediction_result = await asyncio.get_event_loop().run_in_executor(
    #                 None,
    #                 self.price_predictor.add_data_point,
    #                 data
    #             )
                
    #             if prediction_result is not None:
    #                 # Create prediction message
    #                 # Convert numpy values to Python native types
    #                 top_features = {
    #                     str(k): float(v) 
    #                     for k, v in prediction_result['top_features'].items()
    #                 }
                    
    #                 prediction_message = {
    #                     "type": "price_prediction",
    #                     "run_id": run_id,
    #                     "timestamp": datetime.now().isoformat(),
    #                     "prediction": float(prediction_result['prediction']),
    #                     "confidence": float(prediction_result['confidence']),
    #                     "is_confident": bool(prediction_result['is_confident']),
    #                     "current_price": float(data.get('Price', 0)),
    #                     "top_features": top_features
    #                 }
                    
    #                 # Send prediction to MT5
    #                 await self.send_to_mt5(json.dumps(prediction_message))
    #                 logging.info(f"Sent prediction to MT5: {prediction_result['prediction']:.4f} "
    #                         f"(confidence: {prediction_result['confidence']:.4f})")

            
    #         except Exception as pred_error:
    #             logging.error(f"Prediction failed: {pred_error}")
    #             logging.exception(pred_error)
            
    #     except Exception as e:
    #         logging.error(f"Error handling all details data: {e}")
    #         logging.exception(e)


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

    # async def cleanup(self):
    #     logging.info("Cleaning up resources...")
    #     self.socket_send.close()
    #     self.socket_receive.close()
    #     self.signal_socket.close()
    #     self.context.term()


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