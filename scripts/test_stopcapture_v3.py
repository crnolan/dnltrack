#!/usr/bin/env python3

import depthai as dai
import cv2

print("Connecting to device")
device = dai.Device()
with dai.Pipeline(device) as pipeline:
    sockets = device.getConnectedCameras()
    # CAM_B is slower than CAM_C
    cam_b = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=30)
    cam_c = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=30)
    bq = cam_b.requestFullResolutionOutput().createOutputQueue()
    cq = cam_c.requestFullResolutionOutput().createOutputQueue()
    biq = cam_b.inputControl.createInputQueue()
    ciq = cam_c.inputControl.createInputQueue()

    print("Starting pipeline")
    pipeline.start()
    nframes = 0
    ntriggers = 0
    try:
        while pipeline.isRunning():
            b_in = bq.tryGet()
            c_in = cq.tryGet()
            if b_in is not None:
                # Visualizing the frame on slower hosts might have overhead
                cv2.imshow("CamB", b_in.getCvFrame())
            if c_in is not None:
                cv2.imshow("CamC", c_in.getCvFrame())
                # nframes += 1
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("t"):
                ctrl = dai.CameraControl()
                ctrl.setStopStreaming()
                biq.send(ctrl)
                ciq.send(ctrl)
            elif key == ord("s"):
                ctrl = dai.CameraControl()
                ctrl.setStartStreaming()
                biq.send(ctrl)
                ciq.send(ctrl)

    except KeyboardInterrupt:
        print(f"Captured {nframes} frames")