import logging
import time
import sys
import yaml
import multiprocessing
from multiprocessing import Process, Value, Queue, Event
import depthai as dai


def create_pipeline(left_name, right_name, rgb_name, fps, triggered=False):

    def _camera_setup(pipeline, camera, name, fps, triggered):
        if triggered:
            camera.setFps(120)
            camera.initialControl.setExternalTrigger(1, 0)
        else:
            camera.setFps(fps)
        logging.info(f'Setting autoexposure limit to {int(1/(8*fps) * 1e6)} us')
        camera.initialControl.setAutoExposureLimit(int(1/(8*fps) * 1e6)) # microseconds
        record_xout = pipeline.create(dai.node.XLinkOut)
        record_xout.setStreamName(name)
        enc = pipeline.create(dai.node.VideoEncoder)
        enc.setDefaultProfilePreset(
            30, dai.VideoEncoderProperties.Profile.H264_MAIN)
        enc.bitstream.link(record_xout.input)
        return enc, record_xout

    pipeline = dai.Pipeline()

    left = pipeline.create(dai.node.MonoCamera)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    right = pipeline.create(dai.node.MonoCamera)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    rgb = pipeline.create(dai.node.ColorCamera)
    rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)

    left_enc, left_record = _camera_setup(pipeline, left, left_name, fps, triggered)
    right_enc, right_record = _camera_setup(pipeline, right, right_name, fps, triggered)
    rgb_enc, rgb_record = _camera_setup(pipeline, rgb, rgb_name, fps, triggered)

    left.out.link(left_enc.input)
    right.out.link(right_enc.input)
    rgb.video.link(rgb_enc.input)

    # Camera control queues
    mono_ctrl = pipeline.createXLinkIn()
    mono_ctrl.setStreamName(left_name + '_ctrl')
    mono_ctrl.out.link(right.inputControl)
    mono_ctrl.out.link(left.inputControl)

    rgb_ctrl = pipeline.createXLinkIn()
    rgb_ctrl.setStreamName(rgb_name + '_ctrl')
    rgb_ctrl.out.link(rgb.inputControl)

    # Soft trigger
    xtrigger = pipeline.create(dai.node.XLinkIn)
    xtrigger.setStreamName('trigger')
    script = pipeline.create(dai.node.Script)
    xtrigger.out.link(script.inputs['trigger'])
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
        wait_for_trigger = node.io['trigger'].get()
        capture()
        node.warn('Trigger successful')
    """)

    return pipeline, left.getResolutionSize()


def run_capture(ip):
    logger = multiprocessing.get_logger()
    logger.info(f'Capture thread started for device {ip}')
    device_infos = dai.Device.getAllAvailableDevices()

    # Find the camera in the list of available devices
    device_info = None
    for di in device_infos:
        if di.name == ip:
            device_info = di
            break
    if device_info is None:
        raise ValueError(f'Could not find device with IP {ip}')

    logger.info(f'Connecting to {device_info.name}')
    while device_info.state != dai.XLinkDeviceState.X_LINK_BOOTLOADER:
        logger.info(f'Waiting for device {device_info.name} '
                    f'to enter bootloader state')
        time.sleep(1)
    hw_device = dai.Device(device_info)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s [%(threadName)s]')
    multiprocessing.log_to_stderr()

    device_infos = dai.Device.getAllAvailableDevices()
    logging.info(f'Found {len(device_infos)} devices')
    logging.info([dev.name for dev in device_infos])

    device_process = Process(
        target=run_capture,
        args=[device_infos[0].name]
    )
    device_process.start()
    device_process.join()
