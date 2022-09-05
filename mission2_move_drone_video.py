import sys
import traceback
import djitellopy
import av
import cv2  # for avoidance of pylint error
import numpy as np
import time
import os,sys
from threading import Thread
from time import sleep

drone=djitellopy.Tello()


def down(dist):
    global drone
    drone.move_down(dist)

def up(dist):
    global drone
    drone.move_up(dist)

def foward(dist):
    global drone
    drone.move_forward(dist)

def rotate(angle):
    global drone
    drone.rotate_clockwise(angle)


