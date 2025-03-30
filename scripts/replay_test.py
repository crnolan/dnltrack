import depthai as dai
from depthai_sdk import Replay
import cv2
import numpy as np

replay = Replay(r'C:\Users\cnolan\tmp\acan2025test')
pipeline = replay.initPipeline()
left_in, rgb_in, right_in = pipeline.getAllNodes()
depth = pipeline.createStereoDepth()
left_in.out.link(depth.left)
right_in.out.link(depth.right)
# depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
depth.setLeftRightCheck(True)
depth.setExtendedDisparity(True)
depth.setSubpixel(False)
depth.initialConfig.setDisparityShift(95)
topLeft = dai.Point2f(0.2, 0.3)
bottomRight = dai.Point2f(0.8, 1.0)
align = pipeline.create(dai.node.ImageAlign)
left_in.out.link(align.input)
rgb_in.out.link(align.inputAlignTo)
xout_leftaligned = pipeline.create(dai.node.XLinkOut)
xout_leftaligned.setStreamName("leftaligned")
align.outputAligned.link(xout_leftaligned.input)
crop = pipeline.create(dai.node.ImageManip)
crop.initialConfig.setCropRect(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y)
depth.disparity.link(crop.inputImage)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("disparity")
crop.out.link(xout.input)
with dai.Device(pipeline) as device:
    replay.createQueues(device)
    # Output queue will be used to get the disparity frames from the outputs defined above
    q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    qleftalign = device.getOutputQueue(name="leftaligned", maxSize=4, blocking=False)

    while replay.sendFrames():
        # inDisparity = q.get()  # blocking call, will wait until a new data has arrived
        # frame = inDisparity.getFrame()
        # Normalization for better visualization
        # frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

        # cv2.imshow("disparity", frame)

        left = qleftalign.get()
        frame = left.getCvFrame()

        cv2.imshow("left", frame)
        # # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
        # frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        # cv2.imshow("disparity_color", frame)

        if cv2.waitKey(1) == ord('q'):
            break

