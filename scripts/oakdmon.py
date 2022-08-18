#!/usr/bin/env python3
# %%
import depthai as dai
from fractions import Fraction
import time
import sys
import cv2
import numpy as np

# black= np.zeros([200,250,1],dtype="uint8")
# cv2.imshow("Black Image",black)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Start CV2 window thread to display images
# MUST BE DONE BEFORE AV IMPORT, SEE:
# https://github.com/PyAV-Org/PyAV/issues/978
# https://github.com/opencv/opencv/issues/21952
cv2.startWindowThread()
cv2.namedWindow('rgb', cv2.WND_PROP_AUTOSIZE)
cv2.waitKey(1)

import av

fps = 5.
codec = "hevc" # H265 by default
# codec = "h264"
if 2 <= len(sys.argv):
    codec = sys.argv[1].lower()
    if codec == "h265": codec = "hevc"

def get_encoder_profile(codec):
    if codec == "h264": return dai.VideoEncoderProperties.Profile.H264_MAIN
    elif codec == "mjpeg": return dai.VideoEncoderProperties.Profile.MJPEG
    else: return dai.VideoEncoderProperties.Profile.H265_MAIN

# Get an AV codec
avcodec = av.CodecContext.create(codec, "r")

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and output
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setFps(fps)

camLeft = pipeline.create(dai.node.MonoCamera)
camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
# camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
camLeft.setFps(fps)
# camLeft.setStrobeExternal()
camRight = pipeline.create(dai.node.MonoCamera)
camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
# camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
camRight.setFps(fps)

# Camera control (exp, iso, focus)
controlIn = pipeline.createXLinkIn()
controlIn.setStreamName('control')
# controlIn.out.link(camRight.inputControl)
controlIn.out.link(camLeft.inputControl)

# Properties
videoEnc = pipeline.create(dai.node.VideoEncoder)
videoEnc.setDefaultProfilePreset(fps, get_encoder_profile(codec))
# videoEnc.setLossless(True) # Lossless MJPEG, video players usually don't support it
camRgb.video.link(videoEnc.input)
# camMono.out.link(videoEnc.input)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName('enc')
videoEnc.bitstream.link(xout.input)


# Connect to device and start pipeline
# %%
with dai.Device(pipeline) as device:

    print(f"App starting streaming {get_encoder_profile(codec).name} encoded frames into file video.mp4")

    # Control queue used to set camera properties
    controlQueue = device.getInputQueue(controlIn.getStreamName())

    ctrl = dai.CameraControl()
    ctrl.setFrameSyncMode(dai.RawCameraControl.FrameSyncMode.OUTPUT)
    # ctrl.setStrobeSensor(1)
    controlQueue.send(ctrl)

    # device.setIrFloodLightBrightness(300)

    # Output queue will be used to get the encoded data from the output defined above
    q = device.getOutputQueue(name="enc", maxSize=30, blocking=True)

    output_container = av.open('video.mp4', 'w')
    stream = output_container.add_stream(codec, rate=fps)
    stream.time_base = Fraction(1, 1000 * 1000) # Microseconds

    if codec == "mjpeg":
        # We need to set pixel format for MJPEG, for H264/H265 it's yuv420p by default
        stream.pix_fmt = "yuvj420p"

    start = time.time()
    try:
        while True:
            data = q.get().getData() # np.array
            packet = av.Packet(data) # Create new packet with byte array

            # Set frame timestamp
            packet.pts = int((time.time() - start) * 1000 * 1000)

            output_container.mux_one(packet) # Mux the Packet into container

            # Decode the image and display
            frames = avcodec.decode(packet)
            # Retrieve 'bgr' (opencv format) frame
            if len(frames) > 0:
                image = np.array(frames[0].to_image().convert('RGB'))[:, :, ::-1]
                cv2.imshow("rgb", image)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        # Keyboard interrupt (Ctrl + C) detected
        pass

    cv2.destroyAllWindows()
    output_container.close()
