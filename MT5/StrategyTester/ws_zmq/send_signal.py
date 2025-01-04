# import zmq
# import json
# from datetime import datetime

# def send_signal_to_mt5(signal_value):
#     context = zmq.Context()
#     socket = context.socket(zmq.PUSH)
#     socket.connect("tcp://127.0.0.1:5556")  # Connect to MT5's PULL socket
    
#     # Create the message
#     message = {
#         "type": "new_signal",
#         "timestamp": datetime.now().isoformat(),
#         "value": signal_value,
#         "run_id": "python_signal"  # Match your run_id format
#     }
    
#     # Send the message
#     socket.send_string(json.dumps(message))
#     print(f"Sent signal: {message}")
    
#     # Clean up
#     socket.close()
#     context.term()

# # Example usage
# if __name__ == "__main__":
#     # You can call this function whenever you want to send a signal
#     send_signal_to_mt5("EARNINGS: 70% positive")  # Example signal value



# import zmq
# import json
# from datetime import datetime

# def send_signal_to_mt5(signal_value):
#     context = zmq.Context()
#     socket = context.socket(zmq.PUSH)
#     socket.connect("tcp://127.0.0.1:5556")  # Connect to MT5's PULL socket
    
#     # Create the message - signal_value can be any length string
#     message = {
#         "type": "new_signal",
#         "timestamp": datetime.now().isoformat(),
#         "value": str(signal_value),  # Convert to string to ensure JSON compatibility
#         "run_id": "python_signal"
#     }
    
#     # Send the message
#     socket.send_string(json.dumps(message))
#     print(f"Sent signal: {message}")
    
#     # Clean up
#     socket.close()
#     context.term()

# # Example usage
# if __name__ == "__main__":
#     # Example of sending a longer message
#     long_signal = """
#     EARNINGS REPORT:
#     - Revenue: $5.2B (Beat by 8%)
#     - EPS: $2.45 (Beat by 12%)
#     - Guidance: Raised for Q2
#     - Key Metrics: User growth +25% YoY
#     - Management Commentary: Strong momentum in cloud segment
#     """
#     send_signal_to_mt5(long_signal)



import zmq
import json
from datetime import datetime

def send_signal_to_mt5(signal_value):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://127.0.0.1:5558")  # Connect to MT5's PULL socket
    
    # Create the message
    message = {
        "type": "new_signal",
        "timestamp": datetime.now().isoformat(),
        "value": signal_value,
        "run_id": "python_signal"
    }
    
    # Convert to JSON and send
    message_json = json.dumps(message)
    print(f"Sending message: {message_json}")
    socket.send_string(message_json)
    
    # Clean up
    socket.close()
    context.term()

if __name__ == "__main__":
    # Example of sending a longer message
    long_signal = """EARNINGS REPORT:
- Revenue: $5.2B (Beat by 8%)
- EPS: $2.45 (Beat by 12%)
- Guidance: Raised for Q2
- Key Metrics: User growth +25% YoY
- Management Commentary: Strong momentum in cloud segment"""
    
    send_signal_to_mt5(long_signal)
    print("Signal sent successfully")