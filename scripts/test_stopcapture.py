#!/usr/bin/env python3

import cv2
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)

xoutLeft.setStreamName('left')
xoutRight.setStreamName('right')

# Properties
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.OUTPUT)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)

controlIn = pipeline.create(dai.node.XLinkIn)
controlIn.setStreamName('control')
controlIn.out.link(monoRight.inputControl)
controlIn.out.link(monoLeft.inputControl)

# Linking
monoRight.out.link(xoutRight.input)
monoLeft.out.link(xoutLeft.input)

# monoLeft.initialControl.setStopStreaming()
# monoRight.initialControl.setStopStreaming()

device_infos = dai.Device.getAllAvailableDevices()
print(device_infos)

# Connect to device and start pipeline
with dai.Device(device_infos[0]) as device:
    device.startPipeline(pipeline)
    controlQueue = device.getInputQueue('control')

    qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)

    while True:
        inLeft = qLeft.tryGet()
        inRight = qRight.tryGet()

        if inLeft is not None:
            cv2.imshow("left", inLeft.getCvFrame())

        if inRight is not None:
            cv2.imshow("right", inRight.getCvFrame())

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            ctrl = dai.CameraControl()
            ctrl.setStopStreaming()
            controlQueue.send(ctrl)
        elif key == ord('s'):
            ctrl = dai.CameraControl()
            ctrl.setStartStreaming()
            controlQueue.send(ctrl)
