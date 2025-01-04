
import zmq
import time
import json

while True:
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind("tcp://127.0.0.1:5557")
    socket.send_string("Hello from Python")



    socket = context.socket(zmq.PULL)
    socket.bind("tcp://127.0.0.1:5556")
    print(socket.recv_string())
