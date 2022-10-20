import depthai as dai
from fractions import Fraction
import cv2
import time
from datetime import datetime
import numpy as np
import threading
import queue
import logging
import sys
import contextlib

# Start CV2 window thread to display images
# MUST BE DONE BEFORE AV IMPORT, SEE:
# https://github.com/PyAV-Org/PyAV/issues/978
# https://github.com/opencv/opencv/issues/21952
if __name__ == '__main__':
    cv2.startWindowThread()
    cv2.namedWindow('RodentVision', cv2.WND_PROP_AUTOSIZE)
    cv2.waitKey(1)


def create_pipeline(fps, left_name, right_name):
    pipeline = dai.Pipeline()
    stereo = pipeline.create(dai.node.StereoDepth)
    # stereo.setExtendedDisparity(True)
    left = pipeline.create(dai.node.MonoCamera)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    left.setFps(fps)
    right = pipeline.create(dai.node.MonoCamera)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    right.setFps(fps)

    left_enc = pipeline.create(dai.node.VideoEncoder)
    left_enc.setDefaultProfilePreset(
        fps, dai.VideoEncoderProperties.Profile.H264_MAIN)
    right_enc = pipeline.create(dai.node.VideoEncoder)
    right_enc.setDefaultProfilePreset(
        fps, dai.VideoEncoderProperties.Profile.H264_MAIN)

    # left.out.link(left_enc.input)
    # right.out.link(right_enc.input)
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.rectifiedLeft.link(left_enc.input)
    stereo.rectifiedRight.link(right_enc.input)
    left_xout = pipeline.create(dai.node.XLinkOut)
    left_xout.setStreamName(left_name)
    right_xout = pipeline.create(dai.node.XLinkOut)
    right_xout.setStreamName(right_name)
    left_enc.bitstream.link(left_xout.input)
    right_enc.bitstream.link(right_xout.input)
    
    return pipeline


def write_thread(in_q, quit_event, filename, width, height, fps):
    '''Grab encoded images from a queue and save to the provided filename'''
    # setup video context
    import av
    codec = 'h264'
    # codec = 'hevc'
    # codec = 'h264_nvenc'
    output_container = av.open(filename, 'w')
    stream = output_container.add_stream(codec, fps)
    stream.time_base = Fraction(1, 1000*1000) # Milliseconds
    t0 = int(time.time()*1000*1000)
    nbytes = 0
    stream.width = width
    stream.height = height
    logging.info('Start encoding')
    write_count = 0
    while not quit_event.is_set():
        try:
            logging.debug('Writer waiting for data...')
            data = in_q.get(timeout=1)
            ts = int(time.time()*1000*1000) - t0
            logging.debug('Writer got data...')
            write_count += 1
            nbytes += data.shape[0]
            packet = av.Packet(data)
            packet.pts = ts
            packet.dts = ts
            output_container.mux_one(packet)
        except queue.Empty:
            pass
    
    logging.info('Clean up writing')

    # Make sure there are no packets left in the queue
    try:
        logging.info('Writer looking for remaining data...')
        while True:
            data = in_q.get(block=False)
            ts = int(time.time()*1000*1000) - t0
            logging.debug('Writer got extra data...')
            write_count += 1
            nbytes += data.shape[0]
            packet = av.Packet(data)
            packet.pts = ts
            packet.dts = ts
            logging.debug('Muxing packet...')
            output_container.mux_one(packet)
    except queue.Empty:
        pass

    logging.info('Packet count: {}'.format(write_count))
    # Close the file
    output_container.close()
    logging.info('Finished encoding')


def decode_thread(decode_q, display_q, quit_event, name):
    '''Decode images and sent to display queue'''
    logging.info('Starting decode thread {}'.format(name))
    import av
    codec = av.CodecContext.create('h264', 'r')
    while not quit_event.is_set():
        try:
            data = decode_q.get(timeout=1)
            frames = codec.decode(av.Packet(data.copy()))
            if len(frames) > 0:
                image = np.array(frames[0].to_image().convert('RGB'))
                try:
                    display_q.put(image[::2, ::2, ::-1], block=False)
                except queue.Full:
                    logging.warning('Dropped decoded image for {}'.format(name))
        except queue.Empty:
            pass


def capture_thread(device_q, write_q, decode_q, quit_event, name):
    '''Capture images from camera and add to the queue'''

    logging.info('Capture thread started for camera {}'.format(name))
    capture_count = 0
    while not quit_event.is_set():
        data = device_q.get().getData()
        capture_count += 1
        try:
            # If the encoder is not running, this will cause an indefinite
            # thread lock. I should probably have a warning loop on it.
            write_q.put(data)
        except queue.Full:
            logging.warn('Encoding queue full, dropped frame!')
        try:
            decode_q.put(data, block=False)
        except queue.Full:
            logging.warn('Display queue full, showing reduced framerate')
    logging.info('Capture count for camera {}: {}'.format(name, capture_count))


class CameraCapture():
    def __init__(self, device_q, name, fps, width, height):
        self.name = name
        self.filename = '{}.mp4'.format(name)
        self.fps = fps
        self.width  = width
        self.height = height
        self.write_q = queue.Queue(maxsize=fps)
        self.decode_q = queue.Queue(maxsize=1)
        self.display_q = queue.Queue(maxsize=1)
        self.capture_quit = threading.Event()
        self.write_quit = threading.Event()
        self.decode_quit = threading.Event()
        self.write_thread = threading.Thread(
            target=write_thread,
            args=(self.write_q, self.write_quit, self.filename, width, height, fps))
        self.decode_thread = threading.Thread(
            target=decode_thread,
            args=(self.decode_q, self.display_q, self.decode_quit, name))
        self.capture_thread = threading.Thread(
            target=capture_thread,
            args=(device_q, self.write_q, self.decode_q, self.capture_quit, name))

    def start_threads(self):
        logging.info('Starting threads for camera {}...'.format(
            self.name))
        self.write_thread.start()
        self.decode_thread.start()
        self.capture_thread.start()
        while not (self.write_thread.is_alive() and
                   self.decode_thread.is_alive() and
                   self.capture_thread.is_alive()):
            time.sleep(0.1)
        logging.debug('Started threads for camera {}...'.format(
            self.name))

    def stop_threads(self):
        logging.info('Stopping threads for camera {}...'.format(
            self.name))
        self.capture_quit.set()
        while self.capture_thread.is_alive():
            logging.info('Waiting for capture thread to exit...')
            self.capture_thread.join(timeout=0.1)
        self.decode_quit.set()
        self.write_quit.set()
        logging.info('Stopped threads for camera {}...'.format(
            self.name))

if __name__ == '__main__':
    fps = 60
    height = 720
    width = 1280
    # codec = 'h264_nvenc'
    log = logging.getLogger()
    log.setLevel(logging.getLevelName('INFO'))
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s [%(threadName)s] ") # I am printing thread id here
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    log.addHandler(console_handler)

    device_infos = dai.Device.getAllAvailableDevices()
    print(f'Found {len(device_infos)} devices')
    
    with contextlib.ExitStack() as stack:
        openvino_version = dai.OpenVINO.Version.VERSION_2021_4
        queues = {}
        threads = []
        for dev in device_infos:
            time.sleep(1) # Currently required due to XLink race issues
            device: dai.Device = stack.enter_context(
                dai.Device(openvino_version, dev, False))
            device.startPipeline(create_pipeline(fps, 'left', 'right'))
            logging.warn(device.getOutputQueueNames())
            left_q = device.getOutputQueue('left', maxSize=fps, blocking=True)
            right_q = device.getOutputQueue('right', maxSize=fps, blocking=True)
            left = CameraCapture(left_q, 'left', fps, width, height)
            right = CameraCapture(right_q, 'right', fps, width, height)
            left.start_threads()
            right.start_threads()

        try:
            caps = [left, right]
            images = [np.zeros((int(height/2), int(width/2), 3)) for i in range(len(caps))]
            while True:
                changed = False
                for i, cap in enumerate(caps):
                    try:
                        images[i] = cap.display_q.get(timeout=0.001)
                        changed = True
                    except queue.Empty:
                        pass
                if changed:
                    # print('Image sizes: {}x{}x{} and {}x{}x{}'.format(*images[0].shape, *images[1].shape))
                    cv2.imshow('RodentVision', np.concatenate(images, axis=1))
                    cv2.waitKey(1)
        except KeyboardInterrupt:
            for cap in caps:
                try:
                    while True:
                        cap.display_q.get_nowait()
                except queue.Empty:
                    pass
            cv2.destroyAllWindows()

        for cap in caps:
            cap.stop_threads()

    logging.info('Exiting...')
