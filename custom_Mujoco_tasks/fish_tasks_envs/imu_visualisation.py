"""
Displays robot
execute from the same folder: python visualisation_example.py
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
import os
import socket
from timeit import default_timer as timer
import numpy as np
import sys
import time
import random
from datetime import datetime
import re

HOST = 'raspberrypi.local'  # IP address of Raspberry Pi
PORT = 8080  # same arbitrary port as on server
startTime = timer()
cur_time = timer()
max_time = 0
phi = 40
omega = 2*np.pi*1
df_dict = []
df_calib = []
STEPS = 2000
sec_calibration=5
model = load_model_from_path("./fishD.xml")
sim = MjSim(model)

viewer = MjViewer(sim)
modder = TextureModder(sim)
t = 0

try:
    # create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))

    while True:
        s.sendall(str(1500).encode())
        viewer.render()

        # receive response from server
        data = s.recv(1024)
        imu=data.decode()
        df_dict.append([imu])

        model.body_quat()
        t += 1
        if t > 500 and os.getenv('TESTING') is not None:
            break