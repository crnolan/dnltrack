#!/usr/bin/env python

import depthai as dai
import time
import sys
import cv2
import numpy as np

# Start CV2 window thread to display images
# MUST BE DONE BEFORE AV IMPORT, SEE:
# https://github.com/PyAV-Org/PyAV/issues/978
# https://github.com/opencv/opencv/issues/21952
cv2.startWindowThread()
cv2.namedWindow('rgb', cv2.WND_PROP_AUTOSIZE)
cv2.waitKey(1)

import av
av264 = av.CodecContext.create('h264', "r")
av265 = av.CodecContext.create('hevc', 'r')

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
ve1 = pipeline.create(dai.node.VideoEncoder)
ve2 = pipeline.create(dai.node.VideoEncoder)
ve3 = pipeline.create(dai.node.VideoEncoder)

ve1Out = pipeline.create(dai.node.XLinkOut)
ve2Out = pipeline.create(dai.node.XLinkOut)
ve3Out = pipeline.create(dai.node.XLinkOut)

ve1Out.setStreamName('ve1Out')
ve2Out.setStreamName('ve2Out')
ve3Out.setStreamName('ve3Out')

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setFps(30)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setFps(60)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setFps(60)
# Create encoders, one for each camera, consuming the frames and encoding them using H.264 / H.265 encoding
ve1.setDefaultProfilePreset(60, dai.VideoEncoderProperties.Profile.H264_MAIN)
ve2.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.H265_MAIN)
ve3.setDefaultProfilePreset(60, dai.VideoEncoderProperties.Profile.H264_MAIN)

# Linking
monoLeft.out.link(ve1.input)
camRgb.video.link(ve2.input)
monoRight.out.link(ve3.input)
ve1.bitstream.link(ve1Out.input)
ve2.bitstream.link(ve2Out.input)
ve3.bitstream.link(ve3Out.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as dev:

    # Output queues will be used to get the encoded data from the outputs defined above
    outQ1 = dev.getOutputQueue(name='ve1Out', maxSize=30, blocking=True)
    outQ2 = dev.getOutputQueue(name='ve2Out', maxSize=30, blocking=True)
    outQ3 = dev.getOutputQueue(name='ve3Out', maxSize=30, blocking=True)

    # The .h264 / .h265 files are raw stream files (not playable yet)
    nbytes = 0
    start = time.time()
    im1 = np.zeros((720, 1280))
    im2 = np.zeros((720, 1280))
    im3 = np.zeros((720, 1280))
    nframes = []
    with open('mono1.h264', 'wb') as fileMono1H264, open('color.h265', 'wb') as fileColorH265, open('mono2.h264', 'wb') as fileMono2H264:
        print("Press Ctrl+C to stop encoding...")
        while True:
            try:
                nframesper = []
                n = 0
                # Empty each queue
                while outQ1.has():
                    data = outQ1.get().getData()
                    nbytes += data.shape[0]
                    data.tofile(fileMono1H264)
                    n += 1
                    # packet = av.Packet(data) # Create new packet with byte array
                    # frames = av264.decode(packet)
                    # if len(frames) > 0:
                    #     im1 = np.array(frames[0].to_image().convert('RGB'))[:, :, ::-1]
                nframesper.append(n)
                n = 0

                while outQ2.has():
                    data = outQ2.get().getData()
                    nbytes += data.shape[0]
                    data.tofile(fileColorH265)
                    n += 1
                    # packet = av.Packet(data) # Create new packet with byte array
                    # frames = av265.decode(packet)
                    # if len(frames) > 0:
                    #     im2 = np.array(frames[0].to_image().convert('RGB'))[:, :, ::-1]
                    #     im2 = cv2.resize(im2, im1.shape[:2], interpolation=cv2.INTER_LINEAR)
                nframesper.append(n)
                n = 0

                while outQ3.has():
                    data = outQ3.get().getData()
                    nbytes += data.shape[0]
                    data.tofile(fileMono2H264)
                    n += 1
                    # packet = av.Packet(data) # Create new packet with byte array
                    # frames = av264.decode(packet)
                    # if len(frames) > 0:
                    #     im3 = np.array(frames[0].to_image().convert('RGB'))[:, :, ::-1]

                nframesper.append(n)
                if sum(nframesper) > 0:
                    nframes.append(nframesper)
                # cv2.imshow("rgb", np.concatenate([im1, im2, im3], axis=1))
                # cv2.imshow('rgb', im1)
                # cv2.waitKey(1)
                
            except KeyboardInterrupt:
                # Keyboard interrupt (Ctrl + C) detected
                break

    print('Frame counts: {}'.format(nframes))
    print('Average data rate: {}'.format(nbytes / (time.time() - start)))
    print("To view the encoded data, convert the stream file (.h264/.h265) into a video file (.mp4), using commands below:")
    cmd = "ffmpeg -framerate 30 -i {} -c copy {}"
    print(cmd.format("mono1.h264", "mono1.mp4"))
    print(cmd.format("mono2.h264", "mono2.mp4"))
    print(cmd.format("color.h265", "color.mp4"))