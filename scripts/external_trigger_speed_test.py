#!/usr/bin/env python3
# BUG: Slow external trigger readiness and inconsistencies between camera sockets

import depthai as dai

print("Connecting to device")
device = dai.Device()
with dai.Pipeline(device) as pipeline:
    sockets = device.getConnectedCameras()
    # CAM_B is slower than CAM_C
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=60)
    cam.initialControl.setExternalTrigger(1, 0)
    # Without no explicit exposure limit, external triggers are hopelessly slow.
    # With the limit set to 1/120 seconds, no cameras keep up with 25 fps triggering.
    # With the limit set to 1/500 seconds, CAM_C manages 25 fps but CAM_B does not.
    # With the limit set to 1/1000 seconds, CAM_C and CAM_B manage 25 fps.
    cam.initialControl.setAutoExposureLimit(int(1e6/120))
    outputq = cam.requestFullResolutionOutput().createOutputQueue()

    print("Starting pipeline")
    pipeline.start()
    nframes = 0
    ntriggers = 0
    try:
        while pipeline.isRunning():

            videoIn = outputq.get()
            if videoIn is not None:
                nframes += 1

    except KeyboardInterrupt:
        print(f"Captured {nframes} frames")