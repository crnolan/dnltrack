import depthai as dai
import yaml
from fractions import Fraction
import math
import cv2
import time
from datetime import datetime, timedelta
import numpy as np
import threading
import queue
import logging
import sys
import os
from multiprocessing import Process, Value, Queue, Event
import multiprocessing

# Start CV2 window thread to display images
# MUST BE DONE BEFORE AV IMPORT, SEE:
# https://github.com/PyAV-Org/PyAV/issues/978
# https://github.com/opencv/opencv/issues/21952
if __name__ == '__main__':
    os.environ["DEPTHAI_WATCHDOG_INITIAL_DELAY"] = "60000"
    os.environ["DEPTHAI_BOOTUP_TIMEOUT"] = "60000"
    cv2.startWindowThread()
    # cv2.namedWindow('RodentVision', cv2.WND_PROP_AUTOSIZE)
    cv2.namedWindow('RodentVision', cv2.WINDOW_NORMAL)
    cv2.waitKey(1)

import av

camera_map = {
    0: 'rgb',
    1: 'left',
    2: 'right'
}


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


def open_container(name, codec, width, height, fps):
    time_format = '%y%m%d_%H%M%S'
    filename = '{}-{}.mp4'.format(name, time.strftime(time_format))
    output_container = av.open(filename, 'w')
    stream = output_container.add_stream(codec, fps)
    stream.time_base = Fraction(1, 1000*1000)  # Microseconds
    logging.debug('Timebase == {}'.format(stream.time_base))
    stream.width = width
    stream.height = height
    return output_container


def run_capture(ip, filename_root,
                triggered, encodec, fps,
                quit_event, record_event, decode_q,
                camera_select, trigger_event,
                device_state):
    '''Capture images from camera and add to the queue'''
    logger = multiprocessing.get_logger()
    logger.debug(f'Capture thread started for device {ip}')

    # Find the camera in the list of available devices, poll every second
    device_info = None
    while not quit_event.is_set() and device_info is None:
        device_infos = dai.Device.getAllAvailableDevices()
        for di in device_infos:
            if di.name == ip:
                device_info = di
                break
        if device_info is None:
            logger.info(f'Waiting for device {ip} to become available')
            device_state.value = 0
            time.sleep(1)

    if quit_event.is_set():
        logger.info('Capture thread quitting')
        return

    logger.info(f'Connecting to {device_info.name}')
    device_state.value = 1
    while device_info.state != dai.XLinkDeviceState.X_LINK_BOOTLOADER:
        logger.info(f'Waiting for device {device_info.name} '
                    f'to enter bootloader state')
        time.sleep(1)
    hw_device = dai.Device(device_info)
    sn = [filename_root + s for s in ['_left', '_right', '_rgb']]
    logger.debug(f'{sn}')
    logger.info(f'Connected to {device_info.name}'
                f' creating pipeline with triggered == {triggered}')
    device_state.value = 2
    if triggered:
        pipeline, (width, height) = create_pipeline(*sn, fps, True)
    else:
        logger.info(f'Creating pipeline with fps == {fps}')
        pipeline, (width, height) = create_pipeline(*sn, fps, False)
        record_event.set()
    hw_device.setIrFloodLightIntensity(0.2)

    hw_device.startPipeline(pipeline)
    mono_control_q = hw_device.getInputQueue(
        filename_root + '_left_ctrl')
    rgb_control_q = hw_device.getInputQueue(
        filename_root + '_rgb_ctrl')
    trigger_q = hw_device.getInputQueue('trigger')
    streams = hw_device.getOutputQueueNames()
    capture_qs = {
        name: hw_device.getOutputQueue(name=name, maxSize=30, blocking=False)
        for name in streams
    }
    # Open a container for each stream
    containers = {name: open_container(name, encodec, width,
                                       height, fps)
                  for name in streams}
    logger.debug(f'Capture process for device {filename_root} alive')
    write_count = {name: 0 for name in streams}
    capture_count = {name: 0 for name in streams}

    t0 = -1
    treport = time.time()
    device_state.value = 3
    while not quit_event.is_set():
        if time.time() - treport > 10:
            logger.debug(f'Capture process for device {filename_root} '
                         f'alive')
            treport = time.time()

        if trigger_event.is_set():
            buffer = dai.Buffer()
            buffer.setData([1])
            trigger_q.send(buffer)
            trigger_event.clear()

        for name in streams:
            message = capture_qs[name].tryGet()
            if message is None:
                continue
            data = message.getData()
            capture_count[name] += 1
            if camera_map[camera_select.value] == name.split('_')[-1]:
                try:
                    decode_q.put(data, block=False)
                except queue.Full:
                    logger.debug('Decode queue full, showing reduced '
                                 'framerate')
            if record_event.is_set():
                if t0 == -1:
                    t0 = message.getTimestamp()
                ts = message.getTimestamp() - t0
                packet = av.Packet(data)
                packet.pts = ts // timedelta(microseconds=1)
                packet.dts = ts // timedelta(microseconds=1)
                containers[name].mux_one(packet)
                write_count[name] += 1
        time.sleep(0.001)

    if not triggered and record_event.is_set():
        logger.info('Stopping camera streaming')
        ctrl = dai.CameraControl()
        ctrl.setStopStreaming()
        rgb_control_q.send(ctrl)
        mono_control_q.send(ctrl)

    if record_event.is_set():
        logger.info('Writing remaining packets')
        for name in streams:
            message = capture_qs[name].tryGet()
            while message is not None:
                data = message.getData()
                capture_count[name] += 1
                if t0 == -1:
                    t0 = message.getTimestamp()
                ts = message.getTimestamp() - t0
                packet = av.Packet(data)
                packet.pts = ts // timedelta(microseconds=1)
                packet.dts = ts // timedelta(microseconds=1)
                containers[name].mux_one(packet)
                write_count[name] += 1
                message = capture_qs[name].tryGet()

    logger.info(f'Closing device {filename_root}')
    hw_device.close()

    for name in streams:
        logger.info('Capture count for camera {}: {}'.format(name, capture_count[name]))
        logger.info('Write count for camera {}: {}'.format(name, write_count[name]))


def run_decode(decode_q, display_q, quit_event, name, codec):
    '''Decode images and sent to display queue'''
    logger = multiprocessing.get_logger()
    codec = av.CodecContext.create(codec, 'r')
    decode_count = 0
    while not quit_event.is_set():
        try:
            data = decode_q.get(timeout=1)
            frames = codec.decode(av.Packet(data.copy()))
            if len(frames) > 0:
                image = np.array(frames[0].to_image().convert('RGB'))
                decode_count += 1
                try:
                    display_q.put(image[:, :, ::-1], block=False)
                except queue.Full:
                    logger.debug('Display queue full for {}'.format(name))
        except queue.Empty:
            pass
    logger.info(f'Decode count for device {name}: {decode_count}')


class DeviceProxy():
    def __init__(self, ip, group, name, fps, width, height, triggered):
        self.ip = ip
        self.group = group
        self.name = name
        self.fps = fps
        self.width = width
        self.height = height
        self.triggered = triggered
        self.encodec = 'h264'
        self.decodec = 'h264'
        self.filename_root = f'group-{group}_camera-{name}'
        self.camera_select = Value('B', 0)
        self.decode_quit = Event()
        self.decode_q = Queue(maxsize=1)
        self.display_q = Queue(maxsize=1)
        self.record_event = Event()
        self.capture_quit = Event()
        self.trigger_event = Event()
        self.decode_process = None
        self.capture_process = None
        self.device_state = Value('b', -1)

    def start(self):
        if not self.is_decode_alive():
            self.decode_process = Process(
                target=run_decode,
                args=(self.decode_q, self.display_q, self.decode_quit,
                      self.filename_root, self.decodec))
            logging.debug(f'Starting decode process for device '
                          f'{self.filename_root}')
            self.decode_process.start()
            logging.debug(f'Started decode process for device '
                          f'{self.filename_root}')
        if not self.is_capture_alive():
            self.capture_process = Process(
                target=run_capture,
                args=[self.ip, self.filename_root,
                      self.triggered, self.encodec, self.fps,
                      self.capture_quit, self.record_event, self.decode_q,
                      self.camera_select, self.trigger_event,
                      self.device_state])
            logging.debug(f'Starting capture thread for device '
                          f'{self.filename_root}')
            self.capture_process.start()
            logging.debug(f'Started capture thread for device '
                          f'{self.filename_root}')

    def is_capture_alive(self):
        return (self.capture_process is not None
                and self.capture_process.is_alive())

    def is_decode_alive(self):
        return (self.decode_process is not None
                and self.decode_process.is_alive())

    def stop(self):
        self.capture_quit.set()
        self.decode_quit.set()
        while (self.capture_process is not None
               and self.capture_process.is_alive()):
            logging.info(f'Waiting for capture thread to exit for '
                         f'device {self.filename_root}')
            self.capture_process.join(5)
        while not self.display_q.empty():
            self.display_q.get()
        while (self.decode_process is not None
               and self.decode_process.is_alive()):
            logging.info(f'Waiting for decode process to exit for '
                         f'device {self.filename_root}')
            self.decode_process.join(5)
        logging.debug(f'Stopped processes for device {self.filename_root}')

    def get_device_state(self):
        return self.device_state.value

    def is_connected(self):
        return self.device_state.value == 3

    def select_camera(self, value):
        self.camera_select.value = value

    def get_selected_camera(self):
        return self.camera_select.value

    def trigger(self):
        if self.trigger_event.is_set():
            logging.debug(f'Trigger event already set for device {self.filename_root}')
        self.trigger_event.set()

    def enable_recording(self):
        logging.info(f'Enabling recording for device {self.filename_root}')
        self.record_event.set()

    def disable_recording(self):
        self.record_event.clear()

    def is_recording(self):
        return self.record_event.is_set()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s [%(threadName)s]')
    multiprocessing.log_to_stderr()

    device_infos = dai.Device.getAllAvailableDevices()
    logging.info(f'Found {len(device_infos)} devices')
    logging.info([dev.name for dev in device_infos])

    if len(sys.argv) > 1:
        config_fn = sys.argv[1]
    else:
        config_fn = 'config.yaml'
    # If there is a config file, load it
    with open(config_fn, 'r') as f:
        config = yaml.safe_load(f)

    if 'groups' not in config:
        logging.error('No camera groups found in config file')
        sys.exit(1)
    if 'fps' not in config:
        logging.error('No fps found in config file')
        sys.exit(1)
    if 'triggered' not in config:
        logging.error('No trigger information found in config file')
        sys.exit(1)
    else:
        print(f'Triggered == {config["triggered"]}')

    width = 1280
    height = 800
    devices = []
    for group_name, cameras in config['groups'].items():
        for camera_name, details in cameras.items():
            devices.append({
                'device': DeviceProxy(details['ip'], group_name, camera_name,
                                      config['fps'], width, height,
                                      config['triggered']),
                'last_state': -2,
                'image': np.zeros((800, 1280, 3))
            })

    key = None
    n_streams = len(devices)
    grid_w = int(np.ceil(np.sqrt(n_streams)))
    grid_h = int(np.ceil(n_streams / grid_w))
    aspect = (1280*grid_w)/(800*grid_h)
    image_grid = np.arange(grid_w * grid_h)
    image_grid[n_streams:] = -1
    image_grid = image_grid.reshape((grid_h, grid_w))
    disp_im = np.concatenate([np.concatenate([devices[i]['image'] for i in row], axis=1)
                              for row in image_grid], axis=0)
    _, _, winw, winh = cv2.getWindowImageRect('RodentVision')
    w = min(winw, int(aspect * winh))
    h = min(winh, int(winw / aspect))
    disp_im = cv2.resize(disp_im, (w, h), interpolation=cv2.INTER_AREA)
    cv2.resizeWindow('RodentVision', w, h)
    cv2.imshow('RodentVision', disp_im)

    try:
        key = None
        recording = not config['triggered']
        while True:
            changed = False
            if key == ord('q'):
                raise KeyboardInterrupt()
            elif key == ord('0'):
                for d in devices:
                    d['device'].select_camera(0)
            elif key == ord('1'):
                for d in devices:
                    d['device'].select_camera(1)
            elif key == ord('2'):
                for d in devices:
                    d['device'].select_camera(2)
            elif key == ord(' '):
                if not recording:
                    for d in devices:
                        d['device'].trigger()
            elif key == ord('r'):
                recording = True
                for d in devices:
                    if d['device'].triggered:
                        d['device'].enable_recording()
                changed = True
            elif key == ord('s'):
                recording = False
                for d in devices:
                    if d['device'].triggered:
                        d['device'].disable_recording()
                changed = True
            for d in devices:
                device_state = d['device'].get_device_state()
                if device_state != d['last_state']:
                    d['last_state'] = device_state
                    if device_state == -1:
                        d['device'].start()
                        logging.info(f'Starting process for {d["device"].filename_root}')
                        image = np.zeros((800, 1280, 3))
                        cv2.putText(image, 'Initialising', (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2)
                        d['image'] = image
                    if d['last_state'] == 0:
                        logging.info(f'Device {d["device"].filename_root} not connected')
                        image = np.zeros((800, 1280, 3))
                        cv2.putText(image, 'Device not found', (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2)
                        d['image'] = image
                    elif d['last_state'] == 1:
                        logging.info(f'Device {d["device"].filename_root} in bootloader')
                        image = np.zeros((800, 1280, 3))
                        cv2.putText(image, 'Waiting to connect', (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2)
                        d['image'] = image
                    elif d['last_state'] == 2:
                        logging.info(f'Device {d["device"].filename_root} starting')
                        image = np.zeros((800, 1280, 3))
                        cv2.putText(image, 'Starting...', (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2)
                        d['image'] = image
                    elif d['last_state'] == 3:
                        logging.info(f'Device {d["device"].filename_root} connected')
                        image = np.zeros((800, 1280, 3))
                        cv2.putText(image, 'Connected', (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2)
                        d['image'] = image
                    changed = True
                if d['device'].is_connected():
                    try:
                        frame = d['device'].display_q.get_nowait()
                        d['image'] = frame
                        changed = True
                    except queue.Empty:
                        pass
            if changed:
                tchanged = time.time()
                images = []
                for d in devices:
                    image = d['image']
                    cv2.putText(image, f'{d["device"].group}-{d["device"].name}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if d['device'].is_recording():
                        cv2.putText(image, 'Recording', (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    images.append(image)
                tdraw = time.time()
                disp_im = np.concatenate([np.concatenate([images[i] for i in row], axis=1)
                                            for row in image_grid], axis=0)
                tconcat = time.time()
                _, _, winw, winh = cv2.getWindowImageRect('RodentVision')
                w = min(winw, int(aspect * winh))
                h = min(winh, int(winw / aspect))
                disp_im = cv2.resize(disp_im, (w, h), interpolation=cv2.INTER_AREA)
                cv2.resizeWindow('RodentVision', w, h)
                cv2.imshow('RodentVision', disp_im)
                logging.debug(f'Time to draw == {tdraw - tchanged}, '
                                f'time to concat == {tconcat - tdraw}, '
                                f'time to display == {time.time() - tconcat}')
            key = cv2.waitKey(1)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    except Exception as e:
        logging.error(e)
    finally:
        for d in devices:
            d['device'].stop()
    logging.info('Exiting...')
