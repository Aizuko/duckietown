#!/usr/bin/env python3
import cv2
import numpy as np
from time import sleep

def gst_pipeline_string():
    # Parameters from the camera_node
    # Refer here : https://github.com/duckietown/dt-duckiebot-interface/blob/daffy/packages/camera_driver/config/jetson_nano_camera_node/duckiebot.yaml
    res_w, res_h, fps = 640, 480, 30
    fov = 'full'
    camera_mode = 3  # Tried all sensor modes \in [1, 5]

    gst_pipeline = f'nvarguscamerasrc sensor-mode={camera_mode} exposuretimerange="100000 80000000" ! video/x-raw(memory:NVMM), width={res_w}, height={res_h}, format=NV12, framerate={fps}/1 ! nvjpegenc ! appsink'

    #'nvarguscamerasrc ! sensor-mode=3, exposuretimerange="100000 80000000" ! video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! nvjpegenc ! appsink'

    #gst_pipeline = 'nvarguscamerasrc sensor-mode=3 exposuretimerange="100000 80000000" ! video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! appsink '
    #gst_pipeline = """ \
    #    nvarguscamerasrc \
    #    sensor-mode={} exposuretimerange="{} {}" ! \
    #    video/x-raw(memory:NVMM), width={}, height={}, format=NV12, framerate={}/1 ! \
    #    nvvidconv !
    #    video/x-raw, format=BGRx !
    #    videoconvert ! \
    #    appsink \
    #""".format(
    #    3, "100000", "80000000", 640, 480, 30
    #)

    # ---
    print("Using GST pipeline A: `{}`".format(gst_pipeline))
    return gst_pipeline

print("Getting capture device")
cap = cv2.VideoCapture()
print("Opening camera")
e = gst_pipeline_string()

if not cap.open(e, cv2.CAP_GSTREAMER):
    print("==================")
    print("~~~ Open returned false :: probably not working ~~~")
    print("==================")

print("Entering with cv")

while(True):
    print("==== Top of the while ====")
    # Capture frame-by-frame
    ret, frame = cap.read()
    print(ret, frame)  # If this prints None, False, the camera failed to open

    sleep(1)
