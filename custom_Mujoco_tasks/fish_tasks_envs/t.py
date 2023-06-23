import os
import mujoco_py
import numpy as np
from numpy import float64
from mujoco_py import load_model_from_path, MjSim, MjViewer
import time
import mediapy as media
import cv2
import sys
from dm_control import mujoco
import PIL.Image

os.environ['MUJOCO_GL'] = 'egl'
physics = mujoco.Physics.from_xml_path('./fishD.xml')
pixels = physics.render()
PIL.Image.fromarray(pixels)
# Load the model from XML file
model = mujoco.MjModel.from_xml_path('./fishD.xml')
data = mujoco.MjData(model)
# Make renderer, render and show the pixels
# mujoco.
renderer = mujoco.Renderer(model)
while True:
    cv2.imshow('imu',renderer.render())
    cv2.waitKey(0)
    sys.exit()  # to exit from all the processes

cv2.destroyAllWindows()  # destroy all windows

'''
# Load the Mujoco XML model
model = mujoco_py.load_model_from_path("fishD.xml")

# #VAR2
# Create the Mujoco-py simulation
sim = mujoco_py.MjSim(model)
view = False#True
if view:
    viewer = MjViewer(sim)
    viewer.render()
    time.sleep(1)
# Get the ID of the body you want to modify
body_id = sim.model.body_name2id("torso")

# Get the current orientation of the body
orientation = sim.data.body_xquat[body_id]
print(sim.data.body_xquat[body_id])
# sim.data.writable = True
sim.data.body_xquat[body_id] = np.array([0,np.pi,0,1])
sim_state = sim.get_state()
# sim.forward()
# Step the simulation to see the changes
# sim.step()
if view:
    viewer.render()
time.sleep(1)
print(sim.data.body_xquat[body_id])'''