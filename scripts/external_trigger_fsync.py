#!/usr/bin/env python3
import depthai as dai
import cv2
import time

pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
# camRgb.setIspScale(1, 2)
camRgb.initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
camRgb.setFps(120)
camRgb.initialControl.setExternalTrigger(1, 0)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("color")
camRgb.isp.link(xoutRgb.input)

monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoLeft.initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
monoLeft.setFps(120)
monoLeft.initialControl.setExternalTrigger(1, 0)

xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutLeft.setStreamName("left")
monoLeft.out.link(xoutLeft.input)

monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoRight.initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
monoRight.setFps(120)
monoRight.initialControl.setExternalTrigger(1, 0)

xoutRight = pipeline.create(dai.node.XLinkOut)
xoutRight.setStreamName("right")
monoRight.out.link(xoutRight.input)

# Connect to device with pipeline
with dai.Device(pipeline) as device:
    arr = ['left', 'right', 'color']
    # arr = ['color']
    queues = {}
    frames = {}

    for name in arr:
        queues[name] = device.getOutputQueue(name)

    print("Starting...")
    start_time = time.time()
    frame_count = 0

    while True:
        for name in arr:
            if queues[name].has():
                frames[name] = queues[name].get().getCvFrame()
                if name == 'color':
                    frame_count += 1
                    if frame_count % 100 == 0:
                        print(f"Frames captured: {frame_count} ({frame_count / (time.time() - start_time):.2f} fps)")

        for name, frame in frames.items():
            cv2.imshow(name, frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
