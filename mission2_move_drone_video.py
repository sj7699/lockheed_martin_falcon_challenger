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

