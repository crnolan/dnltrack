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
from multiprocessing import Process, Value, Queue, Event

# Start CV2 window thread to display images
# MUST BE DONE BEFORE AV IMPORT, SEE:
# https://github.com/PyAV-Org/PyAV/issues/978
# https://github.com/opencv/opencv/issues/21952
if __name__ == '__main__':
    cv2.startWindowThread()
    # cv2.namedWindow('RodentVision', cv2.WND_PROP_AUTOSIZE)
    cv2.namedWindow('RodentVision', cv2.WINDOW_NORMAL)
    cv2.waitKey(1)

import av


def create_pipeline(left_name, right_name, rgb_name):

    def _camera_setup(pipeline, camera, name):
        camera.setFps(120)
        camera.initialControl.setExternalTrigger(1, 0)
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

    left_enc, left_record = _camera_setup(pipeline, left, left_name)
    right_enc, right_record = _camera_setup(pipeline, right, right_name)
    rgb_enc, rgb_record = _camera_setup(pipeline, rgb, rgb_name)

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


def run_capture(hw_device, decode_q, quit_event, decode_event, record_event,
                name, codec, width, height, fps):
    '''Capture images from camera and add to the queue'''

    logging.debug('Capture thread started for camera {}'.format(name))
    # codec = 'h264'
    # codec = 'hevc'
    # codec = 'h264_nvenc'
    device_q = hw_device.getOutputQueue(name, maxSize=30, blocking=True)
    time_format = '%y%m%d_%H%M%S'
    filename = '{}-{}.mp4'.format(name, time.strftime(time_format))
    output_container = av.open(filename, 'w')
    stream = output_container.add_stream(codec, fps)
    logging.debug('Capture thread for camera {} alive'.format(name))
    stream.time_base = Fraction(1, 1000*1000) # Microseconds
    logging.debug('Timebase == {}'.format(stream.time_base))
    # t0 = int(time.time_ns())
    stream.width = width
    stream.height = height
    write_count = 0
    capture_count = 0

    # t0 = int(time.time_ns())
    t0 = -1
    treport = time.time()
    while not quit_event.is_set():
        if time.time() - treport > 10:
            logging.debug('Capture thread for camera {} alive'.format(name))
            treport = time.time()
        message = device_q.tryGet()
        if message is None:
            time.sleep(0.001)
            continue
        data = message.getData()
        capture_count += 1
        if decode_event.is_set():
            try:
                decode_q.put(data, block=False)
            except queue.Full:
                logging.debug('Decode queue full, showing reduced framerate')
        if record_event.is_set():
            if t0 == -1:
                t0 = message.getTimestamp()
            ts = message.getTimestamp() - t0
            packet = av.Packet(data)
            packet.pts = ts // timedelta(microseconds=1)
            packet.dts = ts // timedelta(microseconds=1)
            # logging.debug('Writing pts / dts == {} at time == {}'.format(ts, time.time_ns()))
            output_container.mux_one(packet)
            write_count += 1
    logging.debug('Capture count for camera {}: {}'.format(name, capture_count))
    logging.info('Write count for camera {}: {}'.format(name, write_count))


def run_decode(decode_q, display_q, quit_event, name, codec):
    '''Decode images and sent to display queue'''
    codec = av.CodecContext.create(codec, 'r')
    while not quit_event.is_set():
        try:
            data = decode_q.get(timeout=1)
            frames = codec.decode(av.Packet(data.copy()))
            if len(frames) > 0:
                image = np.array(frames[0].to_image().convert('RGB'))
                try:
                    display_q.put(image[:, :, ::-1], block=False)
                except queue.Full:
                    logging.debug('Display queue full for {}'.format(name))
        except queue.Empty:
            pass


class CameraCapture():
    def __init__(self, device_q, decode_q, name, resolution, fps, encodec):
        self.name = name
        self.width = resolution[0]
        self.height = resolution[1]
        self.fps = fps
        self.encodec = encodec
        self.decode_q = decode_q
        self.decode = threading.Event()
        self.record = threading.Event()
        self.capture_quit = threading.Event()
        self.capture_process = threading.Thread(
            target=run_capture,
            args=(device_q, self.decode_q, self.capture_quit,
                  self.decode, self.record, name,
                  encodec, self.width, self.height, self.fps))

    def start(self):
        logging.debug('Starting capture process for camera {}...'.format(
            self.name))
        self.capture_process.start()
        # while not (self.write_thread.is_alive() and
        #            self.decode_thread.is_alive() and
        #            self.capture_thread.is_alive()):
        #     time.sleep(0.1)
        logging.debug('Started capture process for camera {}...'.format(
            self.name))

    def enable_decoding(self):
        logging.debug('Enable frame decoding for camera {}...'.format(
            self.name
        ))
        self.decode.set()

    def disable_decoding(self):
        logging.debug('Disable frame decoding for camera {}...'.format(
            self.name
        ))
        self.decode.clear()

    def enable_recording(self):
        logging.debug('Enable frame recording for camera {}...'.format(
            self.name
        ))
        self.record.set()

    def disable_recording(self):
        logging.debug('Disable frame recording for camera {}...'.format(
            self.name
        ))
        self.record.clear()

    def stop(self):
        logging.debug('Stopping processes for camera {}...'.format(
            self.name))
        self.capture_quit.set()
        logging.debug('Waiting for capture thread to exit...')
        self.capture_process.join()


def connect_thread(device):
    logging.info(f'Connecting to {device.device_info.name}')
    hw_device = dai.Device(device.device_info)
    sn = [device.filename_root + s for s in ['_left', '_right', '_rgb']]
    logging.debug(f'{sn}')
    logging.info(f'Connected to {device.device_info.name}, starting pipeline')
    pipeline, resolution = create_pipeline(*sn)
    hw_device.setIrFloodLightIntensity(0.1)
    hw_device.startPipeline(pipeline)
    logging.info(f'Pipeline started for  {device.device_info.name}')
    mono_control_q = hw_device.getInputQueue(sn[0] + '_ctrl')
    rgb_control_q = hw_device.getInputQueue(sn[2] + '_ctrl')
    left = CameraCapture(hw_device, device.decode_q,
                         sn[0], resolution, device.fps, device.encodec)
    right = CameraCapture(hw_device, device.decode_q,
                          sn[1], resolution, device.fps, device.encodec)
    rgb = CameraCapture(hw_device, device.decode_q,
                        sn[2], resolution, device.fps, device.encodec)

    with device.lock:
        device.device = hw_device
        device.pipeline = pipeline
        device.width = resolution[0]
        device.height = resolution[1]
        device.mono_control_q = mono_control_q
        device.rgb_control_q = rgb_control_q
        device.left = left
        device.right = right
        device.rgb = rgb
    device.start()


class Device():
    def __init__(self, name, device_info, group, fps=30):
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
        self.lock = threading.Lock()
        self.filename_root = f'group-{group}_camera-{name}'
        self.rgb_control_q = None
        self.mono_control_q = None
        self.camera_select = 'rgb'
        self.rgb_q = None
        self.left_q = None
        self.right_q = None
        self.decode_quit = Event()
        self.decode_q = Queue(maxsize=1)
        self.display_q = Queue(maxsize=1)
        self.decode_process = Process(
            target=run_decode,
            args=(self.decode_q, self.display_q, self.decode_quit,
                  name, self.decodec))
        self.left = None
        self.right = None
        self.rgb = None

    def connect(self):
        if self.device is not None or self.connect_thread is not None:
            return
        self.connect_thread = threading.Thread(
            target=connect_thread, args=[self], daemon=True)
        self.connect_thread.start()

    def is_connecting(self):
        with self.lock:
            return self.connect_thread is not None and self.device is None

    def is_connected(self):
        with self.lock:
            return self.device is not None

    def start(self):
        if self.is_connected():
            with self.lock:
                logging.debug('Starting decode process for device {}...'.format(
                    self.name))
                self.decode_process.start()
                logging.debug('Started decode process for device {}...'.format(
                    self.name))
                self.left.start()
                self.right.start()
                self.rgb.start()
                self.rgb.enable_decoding()

    def stop(self):
        if self.is_connected():
            with self.lock:
                self.left.stop()
                self.right.stop()
                self.rgb.stop()
                self.decode_quit.set()
                while not self.display_q.empty():
                    self.display_q.get()
                while self.decode_process.is_alive():
                    logging.debug('Waiting for decode thread to exit...')
                    self.decode_process.join(5)
                logging.debug('Stopped processes for camera {}...'.format(
                    self.name))
                self.device.close()

    def select_left(self):
        if self.is_connected():
            with self.lock:
                self.right.disable_decoding()
                self.rgb.disable_decoding()
                self.left.enable_decoding()

    def select_right(self):
        if self.is_connected():
            with self.lock:
                self.left.disable_decoding()
                self.rgb.disable_decoding()
                self.right.enable_decoding()

    def select_rgb(self):
        if self.is_connected():
            with self.lock:
                self.left.disable_decoding()
                self.right.disable_decoding()
                self.rgb.enable_decoding()

    def enable_recording(self):
        if self.is_connected():
            with self.lock:
                self.left.enable_recording()
                self.right.enable_recording()
                self.rgb.enable_recording()

    def disable_recording(self):
        if self.is_connected():
            with self.lock:
                self.left.disable_recording()
                self.right.disable_recording()
                self.rgb.disable_recording()


if __name__ == '__main__':
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s [%(threadName)s] ") # I am printing thread id here
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    log.addHandler(console_handler)

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
                device = Device(camera, device_info, name)
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
                elif key == ord('r'):
                    recording = True
                    for d in devices:
                        d['device'].enable_recording()
                    changed = True
                elif key == ord('s'):
                    for d in devices:
                        d['device'].disable_recording()
                    recording = False
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
                    images = []
                    for d in devices:
                        image = d['image']
                        cv2.putText(image, f'{d["group"]}-{d["camera"]}', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if recording:
                            cv2.putText(image, 'Recording', (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        images.append(image)
                    disp_im = np.concatenate([np.concatenate([images[i] for i in row], axis=1)
                                              for row in image_grid], axis=0)
                    _, _, winw, winh = cv2.getWindowImageRect('RodentVision')
                    w = min(winw, int(aspect * winh))
                    h = min(winh, int(winw / aspect))
                    disp_im = cv2.resize(disp_im, (w, h), interpolation=cv2.INTER_AREA)
                    cv2.resizeWindow('RodentVision', w, h)
                    cv2.imshow('RodentVision', disp_im)
                key = cv2.waitKey(1)
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
        except Exception as e:
            logging.error(e)

    except Exception as e:
        logging.exception(e)
    finally:
        for d in devices:
            d['device'].disable_recording()
            d['device'].stop()
    logging.info('Exiting...')
