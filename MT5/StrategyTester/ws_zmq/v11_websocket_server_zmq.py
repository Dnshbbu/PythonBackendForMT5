

import asyncio
import zmq
import zmq.asyncio
import csv
import json
import os
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go

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

    # Dictionary to store headers for each file
    headers = {}

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
                        # Uncomment the following line to see the entire data
                        # print(f"Received data: {data}")
                    except json.JSONDecodeError as e:
                        print(f"[{datetime.now()}] JSONDecodeError: {e}. Message: {message}")
                        continue  # Skip processing this message and continue with the next


                     # Process the tick data here as needed
                    # For example, sending a response back to MT5
                    response_data = {
                        "response": "Acknowledged",
                        "tick": "data"
                    }
                    if socket_send.send_string("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"):
                        print(f"Sent response to MT5: {response_data}")
                    
                    run_id = data.get("run_id")
                    msg_type_raw = data.get("type", "")
                    msg_type = msg_type_raw.strip().lower()
                    msg_content = data.get("msg")

                    if run_id and msg_type and msg_content:

                        if msg_type == 'draw_plot':
                            draw_plot(run_id)
                        else:
                            # Generate the CSV filename
                            csv_filename = f"{run_id}_{msg_type}.csv"
                            
                            # Open the file in append mode
                            with open(csv_filename, 'a', newline='') as csvfile:
                                csv_writer = csv.writer(csvfile)

                                if msg_type == 'priceandequity':
                                    for row in msg_content:
                                        csv_writer.writerow(row.split(','))
                                    # print(f"Appended {len(msg_content)} rows to {csv_filename}")
                                
                                elif msg_type == 'transactions':
                                    print(msg_content)
                                    # Check if we have headers for this file
                                    if csv_filename not in headers:
                                        # If file is empty, write headers
                                        if os.path.getsize(csv_filename) == 0:
                                            headers[csv_filename] = msg_content.split(',')
                                            csv_writer.writerow(headers[csv_filename])
                                            # print(f"Created CSV file: {csv_filename} with headers")
                                        else:
                                            # If file exists but we don't have headers, read the first line
                                            with open(csv_filename, 'r') as f:
                                                headers[csv_filename] = next(csv.reader(f))
                                        continue  # Skip further processing for header row
                                    
                                    # Now process the data
                                    data_row = msg_content.split(',')
                                    if len(data_row) == len(headers[csv_filename]):
                                        csv_writer.writerow(data_row)
                                        # print(f"Appended data row to {csv_filename}: {data_row}")
                                    else:
                                        print(f"Invalid data row format: {data_row}")
                                        print(f"Expected {len(headers[csv_filename])} columns, got {len(data_row)}")
                                        print(f"Headers: {headers[csv_filename]}")

                                


                                elif msg_type in ['transactions_details', 'price_equity_details']:
                                    # Handle other types (assuming these are single row messages)
                                    if os.path.getsize(csv_filename) == 0:
                                        csv_writer.writerow(msg_content.split(','))
                                        print(f"Created CSV file: {csv_filename} with headers")
                                    else:
                                        csv_writer.writerow(msg_content.split(','))
                                        print(f"Appended data row to {csv_filename}")
                                
                                else:
                                    print(f"Unknown message type: '{msg_type}'")
    
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


def draw_plot(run_id):
    # Load equity and price data from CSV file
    equity_price_file_name = run_id+'_priceandequity.csv'
    equity_price_file_path = r'C:\Users\StdUser\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AMD_EquityPrice_2024.01.01.csv'  # Replace with your CSV file path
    df_equity_price = pd.read_csv(equity_price_file_name)

    # Combine Date and Time into a single datetime column for equity and price data
    df_equity_price['DateTime'] = pd.to_datetime(df_equity_price['Date'] + ' ' + df_equity_price['Time'])

    transactions_file_name = run_id+'_transactions.csv'
    # Load transaction data from CSV file
    transaction_file_path = r'C:\Users\StdUser\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AMD_2024.01.01_2024.02.28_Mul_indicators_w_DT_SL_Eq_TP_R_9640.csv'  # Replace with your transaction CSV file path
    df_transactions = pd.read_csv(transactions_file_name)

    # Combine Date and Time into a single datetime column for transaction data
    df_transactions['DateTime'] = pd.to_datetime(df_transactions['Date'] + ' ' + df_transactions['Time'])

    # Plotting with Plotly
    fig = go.Figure()

    # Add trace for Bid Price
    fig.add_trace(go.Scatter(
        x=df_equity_price['DateTime'],
        y=df_equity_price['Price'],
        mode='lines',
        name='Price',
        yaxis='y1',
        hovertemplate='<b>Price</b><br>Date: %{x}<br>Price: %{y}<extra></extra>'
    ))

    # Add trace for Equity
    fig.add_trace(go.Scatter(
        x=df_equity_price['DateTime'],
        y=df_equity_price['Equity'],
        mode='lines',
        name='Equity',
        yaxis='y2',
        hovertemplate='<b>Equity</b><br>Date: %{x}<br>Equity: %{y}<extra></extra>'
    ))

    # Filter Buy and Sell transactions
    df_buy = df_transactions[df_transactions['Type'].str.contains('Buy')]
    df_sell = df_transactions[df_transactions['Type'].str.contains('Sell')]

    # Add Buy signals (Green upward triangle)
    fig.add_trace(go.Scatter(
        x=df_buy['DateTime'],
        y=df_equity_price.loc[df_equity_price['DateTime'].isin(df_buy['DateTime']), 'Price'],  # Match with equity price DateTime
        mode='markers',
        marker=dict(symbol='triangle-up', color='green', size=10),
        name='Buy',
        hovertemplate=(
            '<b>Buy</b><br>Date: %{x}<br>Price: %{y}<br>'
            'MA Score: %{customdata[0]}<br>MACD Score: %{customdata[1]}<br>RSI Score: %{customdata[2]}<br>'
            'Stoch Score: %{customdata[3]}<br>BB Score: %{customdata[4]}<br>ATR Score: %{customdata[5]}<br>'
            'SAR Score: %{customdata[6]}<br>Ichimoku Score: %{customdata[7]}<br>ADX Score: %{customdata[8]}<br>'
            'Volume Score: %{customdata[9]}<br>Total Score: %{customdata[10]}<extra></extra>'
        ),
        customdata=df_buy[['MA Score', 'MACD Score', 'RSI Score', 'Stoch Score', 'BB Score', 'ATR Score', 'SAR Score', 
                        'Ichimoku Score', 'ADX Score', 'Volume Score', 'Total Score']].values
    ))

    # Add Sell signals (Red inverted triangle)
    fig.add_trace(go.Scatter(
        x=df_sell['DateTime'],
        y=df_equity_price.loc[df_equity_price['DateTime'].isin(df_sell['DateTime']), 'Price'],  # Match with equity price DateTime
        mode='markers',
        marker=dict(symbol='triangle-down', color='red', size=10),
        name='Sell',
        hovertemplate=(
            '<b>Sell</b><br>Date: %{x}<br>Price: %{y}<br>'
            'MA Score: %{customdata[0]}<br>MACD Score: %{customdata[1]}<br>RSI Score: %{customdata[2]}<br>'
            'Stoch Score: %{customdata[3]}<br>BB Score: %{customdata[4]}<br>ATR Score: %{customdata[5]}<br>'
            'SAR Score: %{customdata[6]}<br>Ichimoku Score: %{customdata[7]}<br>ADX Score: %{customdata[8]}<br>'
            'Volume Score: %{customdata[9]}<br>Total Score: %{customdata[10]}<extra></extra>'
        ),
        customdata=df_sell[['MA Score', 'MACD Score', 'RSI Score', 'Stoch Score', 'BB Score', 'ATR Score', 'SAR Score', 
                            'Ichimoku Score', 'ADX Score', 'Volume Score', 'Total Score']].values
    ))

    # Update layout to include a secondary y-axis
    fig.update_layout(
        title='Equity and Bid Price with Buy/Sell Signals Over Time',
        xaxis_title='DateTime',
        yaxis_title='Bid Price',
        yaxis2=dict(title='Equity', overlaying='y', side='right'),
        xaxis_rangeslider_visible=True,
        hovermode='x unified'
    )

    # Set x-axis range based on your data
    min_date = df_equity_price['DateTime'].min()
    max_date = df_equity_price['DateTime'].max()
    fig.update_xaxes(range=[min_date, max_date])

    # Save the figure to an HTML file
    output_file_path = run_id+'_output_plot.html'  # Specify the output HTML file name
    fig.write_html(output_file_path)

    print(f"Plot saved as {output_file_path}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting...")
