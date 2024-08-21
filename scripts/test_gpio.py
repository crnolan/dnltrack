import cv2
import depthai as dai
import time

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
rgb = pipeline.create(dai.node.ColorCamera)

# Properties
rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
rgb.setFps(10)
rgb_xout = pipeline.create(dai.node.XLinkOut)
rgb_xout.setStreamName('rgb')
rgb.video.link(rgb_xout.input)

script = pipeline.create(dai.node.Script)
script.setScript(
    '''
    import GPIO

    MX_PIN = 42
    ret = GPIO.setup(MX_PIN, GPIO.OUT, GPIO.PULL_DOWN)
    toggleVal = True
    node.warn('GPIO toggle: ' + str(toggleVal))

    while (True):
        data = node.io['in'].get()  # Wait for a message from the host computer
        toggleVal = not toggleVal
        node.warn('GPIO toggle: ' + str(toggleVal))
        ret = GPIO.write(MX_PIN, toggleVal)  # Toggle the GPIO
    '''
)

gpiotog_xin = pipeline.create(dai.node.XLinkIn)
gpiotog_xin.setStreamName('gpiotog')
gpiotog_xin.out.link(script.inputs['in'])

# end_out = pipeline.create(dai.node.XLinkOut)
# end_out.setStreamName('end')
# script.outputs['end'].link(end_out.input)

device_infos = dai.Device.getAllAvailableDevices()
print(device_infos)

with dai.Device(device_infos[0]) as device:
    device.startPipeline(pipeline)
    gpiotog = device.getInputQueue('gpiotog')
    while True:
        input()
        gpiotog.send(dai.Buffer())
    # device.getOutputQueue('end').get()
    # time.sleep(3)

