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


def create_pipeline(left_name, right_name, rgb_name, fps=None):

    def _camera_setup(pipeline, camera, name, fps=None):
        if fps is None:
            camera.setFps(120)
            camera.initialControl.setExternalTrigger(1, 0)
        else:
            camera.setFps(fps)
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

    left_enc, left_record = _camera_setup(pipeline, left, left_name, fps)
    right_enc, right_record = _camera_setup(pipeline, right, right_name, fps)
    rgb_enc, rgb_record = _camera_setup(pipeline, rgb, rgb_name, fps)

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


def run_capture(device):
    '''Capture images from camera and add to the queue'''

    logging.debug(f'Capture thread started for device {device.filename_root}')
    device.start_pipeline()
    streams = device.device.getOutputQueueNames()
    # Open a container for each stream
    containers = {name: open_container(name, device.encodec, device.width,
                                       device.height, device.fps)
                  for name in streams}
    logging.debug(f'Capture thread for device {device.filename_root} alive')
    write_count = {name: 0 for name in streams}
    capture_count = {name: 0 for name in streams}

    t0 = -1
    treport = time.time()
    while not device.capture_quit.is_set():
        if time.time() - treport > 10:
            logging.debug(f'Capture thread for device {device.filename_root} '
                          f'alive')
            treport = time.time()
        for name in streams:
            message = device.capture_qs[name].tryGet()
            if message is None:
                continue
            data = message.getData()
            capture_count[name] += 1
            if device.camera_select == name.split('_')[-1]:
                try:
                    device.decode_q.put(data, block=False)
                except queue.Full:
                    logging.debug('Decode queue full, showing reduced '
                                  'framerate')
            if device.record_event.is_set():
                if t0 == -1:
                    t0 = message.getTimestamp()
                ts = message.getTimestamp() - t0
                packet = av.Packet(data)
                packet.pts = ts // timedelta(microseconds=1)
                packet.dts = ts // timedelta(microseconds=1)
                containers[name].mux_one(packet)
                write_count[name] += 1
        time.sleep(0.001)

    device.device.close()

    for name in streams:
        logging.debug('Capture count for camera {}: {}'.format(name, capture_count[name]))
        logging.info('Write count for camera {}: {}'.format(name, write_count[name]))


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


def connect_thread(device):
    logging.info(f'Connecting to {device.device_info.name}')
    while device.device_info.state != dai.XLinkDeviceState.X_LINK_BOOTLOADER:
        logging.info(f'Waiting for device {device.device_info.name} '
                     f'to enter bootloader state')
        time.sleep(1)
    hw_device = dai.Device(device.device_info)
    with device.lock:
        device.device = hw_device
    sn = [device.filename_root + s for s in ['_left', '_right', '_rgb']]
    logging.debug(f'{sn}')
    logging.info(f'Connected to {device.device_info.name}'
                 f' creating pipeline with triggered == {device.triggered}')
    if device.triggered:
        pipeline, resolution = create_pipeline(*sn)
    else:
        logging.info(f'Creating pipeline with fps == {device.fps}')
        pipeline, resolution = create_pipeline(*sn, device.fps)
        device.enable_recording()
    hw_device.setIrFloodLightIntensity(0.1)

    with device.lock:
        device.pipeline = pipeline
        device.width = resolution[0]
        device.height = resolution[1]
    device.start()


class Device():
    def __init__(self, name, device_info, group, fps, triggered):
        self.name = name
        self.device_info = device_info
        self.group = group
        self.encodec = 'h264'
        self.decodec = 'h264'
        self.pipeline = None
        self.connect_thread = None
        self.device = None
        self.width = None
        self.height = None
        self.fps = fps
        self.triggered = triggered
        self.lock = threading.Lock()
        self.filename_root = f'group-{group}_camera-{name}'
        self.rgb_control_q = None
        self.mono_control_q = None
        self.camera_select = 'rgb'
        self.decode_quit = Event()
        self.decode_q = Queue(maxsize=1)
        self.display_q = Queue(maxsize=1)
        self.record_event = threading.Event()
        self.capture_quit = threading.Event()
        self.connect_thread = None
        self.decode_process = None
        self.capture_thread = None
        self.capture_qs = {}

    def connect(self):
        if self.device is not None or self.connect_thread is not None:
            return
        self.connect_thread = threading.Thread(            target=connect_thread, args=[self], daemon=True)
        self.connect_thread.start()

    def is_connecting(self):
        return self.connect_thread is not None and self.device is None

    def is_connected(self):
        return self.device is not None

    def is_running(self):
        if not self.is_connected():
            return False
        if self.device.isClosed():
            return False
        if self.device.isPipelineRunning():
            return True
        return False

    def start_pipeline(self):
        if self.is_connected() and not self.is_running():
            self.device.startPipeline(self.pipeline)
            with self.lock:
                self.mono_control_q = self.device.getInputQueue(
                    self.filename_root + '_left_ctrl')
                self.rgb_control_q = self.device.getInputQueue(
                    self.filename_root + '_rgb_ctrl')
                for name in self.device.getOutputQueueNames():
                    self.capture_qs[name] = self.device.getOutputQueue(
                        name=name, maxSize=30, blocking=False)

    def start(self):
        if self.is_connected() and not self.is_running():
            if self.decode_process is None or not self.decode_process.is_alive():
                self.decode_process = Process(
                    target=run_decode,
                    args=(self.decode_q, self.display_q, self.decode_quit,
                          self.filename_root, self.decodec))
                logging.debug(f'Starting decode process for device '
                              f'{self.filename_root}')
                self.decode_process.start()
                logging.debug(f'Started decode process for device '
                              f'{self.filename_root}')
            if self.capture_thread is None or not self.capture_thread.is_alive():
                self.capture_thread = threading.Thread(
                        target=run_capture,
                        args=[self])
                logging.debug(f'Starting capture thread for device '
                              f'{self.filename_root}')
                self.capture_thread.start()
                logging.debug(f'Started capture thread for device '
                              f'{self.filename_root}')

    def stop(self):
        if self.is_connected():
            ctrl = dai.CameraControl()
            ctrl.setStopStreaming()
            self.rgb_control_q.send(ctrl)
            self.mono_control_q.send(ctrl)
            time.sleep(1)
            self.capture_quit.set()
            self.decode_quit.set()

    def close(self):
        if self.is_running():
            self.stop()
        while self.capture_thread.is_alive():
            logging.info(f'Waiting for capture thread to exit for '
                         f'device {self.filename_root}')
            self.capture_thread.join(5)
        while not self.display_q.empty():
            self.display_q.get()
        while self.decode_process.is_alive():
            logging.info(f'Waiting for decode process to exit for '
                         f'device {self.filename_root}')
            self.decode_process.join(5)
        logging.debug(f'Stopped processes for device {self.filename_root}')

    def select_left(self):
        with self.lock:
            self.camera_select = 'left'

    def select_right(self):
        with self.lock:
            self.camera_select = 'right'

    def select_rgb(self):
        with self.lock:
            self.camera_select = 'rgb'

    def enable_recording(self):
        if self.is_connected():
            logging.info(f'Enabling recording for device {self.filename_root}')
            self.record_event.set()

    def disable_recording(self):
        if self.is_connected():
            self.record_event.clear()

    def is_recording(self):
        return self.record_event.is_set()


if __name__ == '__main__':
    # log = logging.getLogger()
    # log.setLevel(logging.INFO)
    # log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s [%(threadName)s] ") # I am printing thread id here
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(log_formatter)
    # log.addHandler(console_handler)
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

    devices = []
    try:
        for name, cameras in config['groups'].items():
            for camera, details in cameras.items():
                # Find the camera in the list of available devices
                device_info = None
                for di in device_infos:
                    if di.name == details['ip']:
                        device_info = di
                        break
                if device_info is None:
                    raise ValueError(f'Could not find device with IP {details["ip"]}')
                device = Device(camera, device_info, name,
                                config['fps'], config['triggered'])
                device.connect()
                devices.append({
                    'group': name,
                    'camera': camera,
                    'device': device,
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
            recording = False
            while True:
                changed = False
                if key == ord('q'):
                    raise KeyboardInterrupt()
                elif key == ord('0'):
                    for d in devices:
                        d['device'].select_rgb()
                elif key == ord('1'):
                    for d in devices:
                        d['device'].select_left()
                elif key == ord('2'):
                    for d in devices:
                        d['device'].select_right()
                elif key == ord('r') and d['device'].triggered:
                    for d in devices:
                        d['device'].enable_recording()
                    changed = True
                elif key == ord('s') and d['device'].triggered:
                    for d in devices:
                        d['device'].disable_recording()
                    changed = True
                for d in devices:
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
                        cv2.putText(image, f'{d["group"]}-{d["camera"]}', (10, 30),
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

    except Exception as e:
        logging.exception(e)
    finally:
        for d in devices:
            d['device'].stop()
        for d in devices:
            d['device'].close()
    logging.info('Exiting...')
