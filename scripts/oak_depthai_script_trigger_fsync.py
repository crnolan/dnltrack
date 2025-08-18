#!/usr/bin/env python3
import depthai as dai
import cv2
import time
import numpy as np

pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setIspScale(2,3)
camRgb.initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
camRgb.setFps(120)
camRgb.initialControl.setExternalTrigger(1, 0)
camRgb.initialControl.setAutoExposureLimit(int(1/(8*25) * 1e6)) # microseconds

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("color")
camRgb.isp.link(xoutRgb.input)

monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoLeft.initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
monoLeft.setFps(120)
monoLeft.initialControl.setExternalTrigger(1, 0)
monoLeft.initialControl.setAutoExposureLimit(int(1/(8*25) * 1e6)) # microseconds

xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutLeft.setStreamName("left")
monoLeft.out.link(xoutLeft.input)

monoRight = pipeline.createMonoCamera()
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoRight.setFps(120)
monoRight.initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
monoRight.initialControl.setExternalTrigger(1, 0)
monoRight.initialControl.setAutoExposureLimit(int(1/(8*25) * 1e6)) # microseconds

xoutRight = pipeline.create(dai.node.XLinkOut)
xoutRight.setStreamName("right")
monoRight.out.link(xoutRight.input)

xin = pipeline.create(dai.node.XLinkIn)
xin.setStreamName('in')
script = pipeline.create(dai.node.Script)
xin.out.link(script.inputs['in'])
script.setScript("""
import GPIO
import time
GPIO_PIN=41 # Trigger

GPIO.setup(GPIO_PIN, GPIO.OUT, GPIO.PULL_DOWN)

def capture():
    GPIO.write(GPIO_PIN, True)
    time.sleep(0.001) # 1ms pulse is enough
    GPIO.write(GPIO_PIN, False)

while True:
    wait_for_trigger = node.io['in'].get()
    capture()
    node.warn('Trigger successful')
""")


xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName('out')
script.outputs['out'].link(xout.input)

# Connect to device with pipeline
with dai.Device(pipeline) as device:
    inQ = device.getInputQueue("in")

    arr = ['left', 'right', 'color']
    queues = {}
    frames = {}

    for name in arr:
        queues[name] = device.getOutputQueue(name)

    def trigger():
        buffer = dai.Buffer()
        buffer.setData([1])
        inQ.send(buffer)

    time.sleep(1)
    t0 = time.time()
    trigger() # Inital trigger
    cv2.imshow('left', np.zeros((800, 1280), dtype=np.uint8))

    while True:
        if (time.time() - t0) > (1./25):
            t0 = time.time()
            trigger()
        for name in arr:
            if queues[name].has():
                frames[name]=queues[name].get().getCvFrame()

        # for name, frame in frames.items():
        #     cv2.imshow(name, frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            trigger()