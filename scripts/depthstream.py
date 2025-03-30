import depthai as dai
import cv2
import numpy as np

pipeline = dai.Pipeline()
cam_left = pipeline.create(dai.node.MonoCamera)
cam_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
cam_left.setFps(30)
cam_right = pipeline.create(dai.node.MonoCamera)
cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
cam_right.setFps(30)

# left_in, rgb_in, right_in = pipeline.getAllNodes()
depth = pipeline.createStereoDepth()
cam_left.out.link(depth.left)
cam_right.out.link(depth.right)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
depth.setLeftRightCheck(True)
depth.setExtendedDisparity(True)
depth.setSubpixel(False)
depth.initialConfig.setDisparityShift(80)
# topLeft = dai.Point2f(0.2, 0.4)
# bottomRight = dai.Point2f(0.8, 1.0)
# crop = pipeline.create(dai.node.ImageManip)
# crop.initialConfig.setCropRect(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y)
# depth.disparity.link(crop.inputImage)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("disparity")
# crop.out.link(xout.input)
depth.disparity.link(xout.input)
with dai.Device(pipeline) as device:
    device.setIrLaserDotProjectorBrightness(500)
    # Output queue will be used to get the disparity frames from the outputs defined above
    q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

    while True:
        inDisparity = q.get()  # blocking call, will wait until a new data has arrived
        frame = inDisparity.getFrame()
        # Normalization for better visualization
        frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

        cv2.imshow("disparity", frame)

        # # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
        # frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        # cv2.imshow("disparity_color", frame)

        if cv2.waitKey(1) == ord('q'):
            break

