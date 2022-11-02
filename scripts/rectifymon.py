#!/usr/bin/env python3
# %%
import depthai as dai
from fractions import Fraction
import time
import sys
import cv2
import numpy as np
from sys import getsizeof

# black= np.zeros([200,250,1],dtype="uint8")
# cv2.imshow("Black Image",black)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Start CV2 window thread to display images
# MUST BE DONE BEFORE AV IMPORT, SEE:
# https://github.com/PyAV-Org/PyAV/issues/978
# https://github.com/opencv/opencv/issues/21952
cv2.startWindowThread()
cv2.namedWindow('left', cv2.WND_PROP_AUTOSIZE)
cv2.namedWindow('right', cv2.WND_PROP_AUTOSIZE)
cv2.waitKey(1)

import av

bwFps = 60.
rgbFps= 60.
# codec = "hevc" # H265 by default
codec = "h264"
if 2 <= len(sys.argv):
    codec = sys.argv[1].lower()
    if codec == "h265": codec = "hevc"

def get_encoder_profile(codec):
    if codec == "h264": return dai.VideoEncoderProperties.Profile.H264_MAIN
    elif codec == "mjpeg": return dai.VideoEncoderProperties.Profile.MJPEG
    else: return dai.VideoEncoderProperties.Profile.H265_MAIN

# Get an AV codec
avcodec_left = av.CodecContext.create('h264', "r")
avcodec_right = av.CodecContext.create('h264', "r")
# avcodec_rgb = av.CodecContext.create('hevc', "r")

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and output
# camRgb = pipeline.create(dai.node.ColorCamera)
# camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
# camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
# camRgb.setFps(rgbFps)
camLeft = pipeline.create(dai.node.MonoCamera)
camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
camLeft.setFps(bwFps)
# camLeft.setStrobeExternal()
camRight = pipeline.create(dai.node.MonoCamera)
camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
camRight.setFps(bwFps)

# Camera control (exp, iso, focus)
controlIn = pipeline.createXLinkIn()
controlIn.setStreamName('control')
# controlIn.out.link(camRight.inputControl)
controlIn.out.link(camLeft.inputControl)

stereo = pipeline.create(dai.node.StereoDepth)

# Better handling for occlusions:
stereo.setLeftRightCheck(False)
# Closer-in minimum depth, disparity range is doubled:
stereo.setExtendedDisparity(True)
# Better accuracy for longer distance, fractional disparity 32-levels:
stereo.setSubpixel(False)

# Define and configure MonoCamera nodes beforehand
camLeft.out.link(stereo.left)
camRight.out.link(stereo.right)

# Properties
leftEnc = pipeline.create(dai.node.VideoEncoder)
leftEnc.setDefaultProfilePreset(bwFps, get_encoder_profile("h264"))
rightEnc = pipeline.create(dai.node.VideoEncoder)
rightEnc.setDefaultProfilePreset(bwFps, get_encoder_profile("h264"))
# rgbEnc = pipeline.create(dai.node.VideoEncoder)
# rgbEnc.setDefaultProfilePreset(rgbFps, get_encoder_profile("hevc"))
# videoEnc.setLossless(True) # Lossless MJPEG, video players usually don't support it
stereo.rectifiedLeft.link(leftEnc.input)
stereo.rectifiedRight.link(rightEnc.input)

leftXOut = pipeline.create(dai.node.XLinkOut)
leftXOut.setStreamName('leftenc')
leftEnc.bitstream.link(leftXOut.input)
rightXOut = pipeline.create(dai.node.XLinkOut)
rightXOut.setStreamName('rightenc')
rightEnc.bitstream.link(rightXOut.input)
# rgbXOut = pipeline.create(dai.node.XLinkOut)
# rgbXOut.setStreamName('rgbenc')
# rgbEnc.bitstream.link(rgbXOut.input)


# Connect to device and start pipeline
# %%
with dai.Device(pipeline) as device:
    print(f'Connected to device {device.getDeviceInfo()}')

    print(f"App starting streaming {get_encoder_profile(codec).name} encoded frames into file video.mp4")

    # Control queue used to set camera properties
    controlQueue = device.getInputQueue(controlIn.getStreamName())
    # rgbControlQueue = device.getInputQueue(rgbControlIn.getStreamName())

    ctrl = dai.CameraControl()
    # ctrl.setFrameSyncMode(dai.RawCameraControl.FrameSyncMode.OUTPUT)
    # ctrl.setStrobeSensor(1)
    # controlQueue.send(ctrl)

    # ctrl = dai.CameraControl()
    # ctrl.setStrobeExternal(41, 1)
    # controlQueue.send(ctrl)

    # ctrl = dai.CameraControl()
    # ctrl.setManualWhiteBalance(1000)
    # rgbControlQueue.send(ctrl)

    # device.setIrFloodLightBrightness(300)

    # Output queue will be used to get the encoded data from the output defined above
    leftQ = device.getOutputQueue(name="leftenc", maxSize=30, blocking=True)
    rightQ = device.getOutputQueue(name="rightenc", maxSize=30, blocking=True)
    # rgbQ = device.getOutputQueue(name="rgbenc", maxSize=30, blocking=True)

    output_container_left = av.open('videoLeft.mp4', 'w')
    output_container_right = av.open('videoRight.mp4', 'w')
    # output_container_rgb = av.open('videoRgb.mp4', 'w')
    timebase = Fraction(1, 1000 * 1000) # Microseconds
    stream_left = output_container_left.add_stream('h264', rate=bwFps)
    stream_left.time_base = timebase
    stream_right = output_container_right.add_stream('h264', rate=bwFps)
    stream_right.time_base = timebase
    # stream_rgb = output_container_rgb.add_stream('h264', rate=rgbFps)
    # stream_rgb.time_base = Fraction(1, 1000 * 1000) # Microseconds

    # if codec == "mjpeg":
    #     # We need to set pixel format for MJPEG, for H264/H265 it's yuv420p by default
    #     stream_rgb.pix_fmt = "yuvj420p"

    print('Starting capture')
    start = time.time()
    nbytes = 0
    imageLeft = np.zeros((360, 640, 3))
    imageRight = np.zeros((360, 640, 3))
    # image = np.zeros((720, 1280, 3))
    try:
        while True:
            ts = int((time.time() - start) * 1000 * 1000)
            leftData = leftQ.get().getData() # np.array
            nbytes += leftData.shape[0]
            rightData = rightQ.get().getData() # np.array
            nbytes += rightData.shape[0]
            # rgbData = rgbQ.get().getData() # np.array
            # print('Got rgb data')
            # nbytes += rgbData.shape[0]
            packet_left = av.Packet(leftData) # Create new packet with byte array
            packet_right = av.Packet(rightData) # Create new packet with byte array
            # packet_rgb = av.Packet(rgbData)

            # Set frame timestamp
            packet_left.pts = ts
            packet_left.dts = ts
            # packet_left.time_base = timebase
            packet_right.pts = ts
            packet_right.dts = ts
            # packet_right.time_base = timebase
            # packet_rgb.pts = int((time.time() - start) * 1000 * 1000)

            output_container_left.mux_one(packet_left) # Mux the Packet into container
            output_container_right.mux_one(packet_right) # Mux the Packet into container
            # output_container_rgb.mux_one(packet_rgb) # Mux the Packet into container

            # Decode the image and display
            frames_left = avcodec_left.decode(av.Packet(leftData.copy()))
            frames_right = avcodec_right.decode(av.Packet(rightData.copy()))
            # Retrieve 'bgr' (opencv format) frame
            # if len(frames_left) > 0:
            #     image = np.concatenate([np.array(frames_left[0].to_image().convert('RGB'))[::2, ::2, ::-1],
            #                             image[360:, :, :]])
            #     # image[360:, :, :] = np.array(frames_left[0].to_image().convert('RGB'))[::2, ::2, ::-1]
            # if len(frames_right) > 0:
            #     image = np.concatenate([image[:360, :, :],
            #                             np.array(frames_right[0].to_image().convert('RGB'))[::2, ::2, ::-1]])
            #     # image[360:, 640:, :] = np.array(frames_right[0].to_image().convert('RGB'))[::2, ::2, ::-1]
            if len(frames_left) > 0:
                imageLeft = np.array(frames_left[0].to_image().convert('RGB'))[::2, ::2, ::-1]
            if len(frames_right) > 0:
                imageRight = np.array(frames_right[0].to_image().convert('RGB'))[::2, ::2, ::-1]
            cv2.imshow("left", imageLeft)
            cv2.imshow("right", imageRight)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        # Keyboard interrupt (Ctrl + C) detected
        pass

    print('Average data rate: {}'.format(nbytes / (time.time() - start)))

    cv2.destroyAllWindows()
    output_container_left.close()
    output_container_right.close()
    # output_container_rgb.close()
