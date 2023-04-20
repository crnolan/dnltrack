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


def create_pipeline(fps, left_name, right_name, rgb_name, disparity_name):
    pipeline = dai.Pipeline()

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setExtendedDisparity(True)
    stereo.setLeftRightCheck(True)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

    # Calculate minZ and maxZ based on FOV and baseline
    # DFOV / HFOV / VFOV: 150°/128°/80°
    # Resolution: 1280x800
    # min_distance = focal_length_in_pixels * base_line_dist / max_disparity_in_pixels
    # focal_length_in_pixels = 1280 * 0.5 / tan(71.9 * 0.5 * PI / 180)
    stereo.initialConfig.setDisparityShift(60)

    left = pipeline.create(dai.node.MonoCamera)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    left.setFps(fps)
    right = pipeline.create(dai.node.MonoCamera)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    right.setFps(fps)

    rgb = pipeline.create(dai.node.ColorCamera)
    rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    # rgb.setVideoSize(640, 360)
    rgb.setFps(fps)


    left_enc = pipeline.create(dai.node.VideoEncoder)
    left_enc.setDefaultProfilePreset(
        fps, dai.VideoEncoderProperties.Profile.H264_MAIN)
    right_enc = pipeline.create(dai.node.VideoEncoder)
    right_enc.setDefaultProfilePreset(
        fps, dai.VideoEncoderProperties.Profile.H264_MAIN)
    rgb_enc = pipeline.create(dai.node.VideoEncoder)
    rgb_enc.setDefaultProfilePreset(
        fps, dai.VideoEncoderProperties.Profile.H264_MAIN)
    disparity_enc = pipeline.create(dai.node.VideoEncoder)
    disparity_enc.setDefaultProfilePreset(
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

    stereo.disparity.link(disparity_enc.input)
    disparity_xout = pipeline.create(dai.node.XLinkOut)
    disparity_xout.setStreamName(disparity_name)
    disparity_enc.bitstream.link(disparity_xout.input)

    rgb.video.link(rgb_enc.input)
    rgb_xout = pipeline.create(dai.node.XLinkOut)
    rgb_xout.setStreamName(rgb_name)
    rgb_xout.input.setBlocking(False)
    rgb_xout.input.setQueueSize(1)
    rgb_enc.bitstream.link(rgb_xout.input)

    # Camera control queue
    controlIn = pipeline.createXLinkIn()
    controlIn.setStreamName('control')
    controlIn.out.link(left.inputControl)

    return pipeline


def write_thread(in_q, quit_event, filename, width, height, fps, codec):
    '''Grab encoded images from a queue and save to the provided filename'''
    # setup video context
    import av
    # codec = 'h264'
    # codec = 'hevc'
    # codec = 'h264_nvenc'
    output_container = av.open(filename, 'w')
    stream = output_container.add_stream(codec, fps)
    stream.time_base = Fraction(1, 1000*1000) # Nanoseconds
    # stream.time_base = Fraction(1, )
    logging.debug('Timebase == {}'.format(stream.time_base))
    # t0 = int(time.time_ns())
    nbytes = 0
    stream.width = width
    stream.height = height
    logging.info('Start writing')
    write_count = 0
    while not quit_event.is_set():
        try:
            logging.debug('Writer waiting for data...')
            data = in_q.get(timeout=1)
            # ts = int(time.time_ns()) - t0
            logging.debug('Writer got data...')
            nbytes += data.shape[0]
            packet = av.Packet(data)
            packet.pts = write_count * 1e6 / fps
            packet.dts = write_count * 1e6 / fps
            # logging.debug('Writing pts / dts == {} at time == {}'.format(ts, time.time_ns()))
            output_container.mux_one(packet)
            write_count += 1
        except queue.Empty:
            pass
    
    logging.info('Clean up writing')

    # Make sure there are no packets left in the queue
    try:
        logging.info('Writer looking for remaining data...')
        while True:
            data = in_q.get(block=False)
            # ts = int(time.time()*1000*1000) - t0
            logging.debug('Writer got extra data...')
            nbytes += data.shape[0]
            packet = av.Packet(data)
            packet.pts = write_count * 1e6 / fps
            packet.dts = write_count * 1e6 / fps
            logging.debug('Muxing packet...')
            output_container.mux_one(packet)
            write_count += 1
    except queue.Empty:
        pass

    logging.info('Packet count: {}'.format(write_count))
    # Close the file
    output_container.close()
    logging.info('Finished encoding')


def decode_thread(decode_q, display_q, quit_event, name, codec):
    '''Decode images and sent to display queue'''
    logging.info('Starting decode thread {}'.format(name))
    import av
    codec = av.CodecContext.create(codec, 'r')
    while not quit_event.is_set():
        try:
            data = decode_q.get(timeout=1)
            frames = codec.decode(av.Packet(data.copy()))
            if len(frames) > 0:
                image = np.array(frames[0].to_image().convert('RGB'))
                try:
                    display_q.put(image[::2, ::2, ::-1], block=False)
                except queue.Full:
                    logging.debug('Display queue full for {}'.format(name))
        except queue.Empty:
            pass


def capture_thread(device_q, write_q, decode_q, quit_event, decode_event, name):
    '''Capture images from camera and add to the queue'''

    logging.info('Capture thread started for camera {}'.format(name))
    capture_count = 0
    # t0 = int(time.time_ns())
    while not quit_event.is_set():
        data = device_q.get().getData()
        # ts = int(time.time_ns()) - t0
        capture_count += 1
        try:
            # If the encoder is not running, this will cause an indefinite
            # thread lock. I should probably have a warning loop on it.
            write_q.put(data)
        except queue.Full:
            logging.warning('Encoding queue full, dropped frame!')
        try:
            if decode_event.is_set():
                decode_q.put(data, block=False)
        except queue.Full:
            logging.info('Display queue full, showing reduced framerate')
    logging.info('Capture count for camera {}: {}'.format(name, capture_count))


class CameraCapture():
    def __init__(self, device_q, name, fps, width, height, decodec, encodec):
        self.name = name
        time_format = '%y%m%d_%H%M%S'
        self.filename = '{}-{}.mp4'.format(name, time.strftime(time_format))
        self.fps = fps
        self.width  = width
        self.height = height
        self.decodec = decodec
        self.encodec = encodec
        self.write_q = queue.Queue(maxsize=fps*10)
        self.decode_q = queue.Queue(maxsize=1)
        self.display_q = queue.Queue(maxsize=1)
        self.capture_quit = threading.Event()
        self.write_quit = threading.Event()
        self.decode_quit = threading.Event()
        self.decode_enable = threading.Event()
        self.write_thread = threading.Thread(
            target=write_thread,
            args=(self.write_q, self.write_quit, self.filename, width,
                  height, fps, encodec))
        self.decode_thread = threading.Thread(
            target=decode_thread,
            args=(self.decode_q, self.display_q, self.decode_quit, name,
                  decodec))
        self.capture_thread = threading.Thread(
            target=capture_thread,
            args=(device_q, self.write_q, self.decode_q, self.capture_quit,
                  self.decode_enable, name))

    def start_threads(self):
        logging.info('Starting threads for camera {}...'.format(
            self.name))
        self.write_thread.start()
        self.decode_thread.start()
        self.capture_thread.start()
        # while not (self.write_thread.is_alive() and
        #            self.decode_thread.is_alive() and
        #            self.capture_thread.is_alive()):
        #     time.sleep(0.1)
        logging.debug('Started threads for camera {}...'.format(
            self.name))

    def enable_decoding(self):
        logging.info('Enable frame decoding for camera {}...'.format(
            self.name
        ))
        self.decode_enable.set()

    def disable_decoding(self):
        logging.info('Disable frame decoding for camera {}...'.format(
            self.name
        ))
        self.decode_enable.clear()

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
    fps = 30
    height = 800
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
    print(device_infos)
    
    with contextlib.ExitStack() as stack:
        openvino_version = dai.OpenVINO.Version.VERSION_2021_4
        caps = []
        disp_caps = []
        ordered_devs = [dai.Device.getDeviceByMxId('18443010C1A4631200'),
                        dai.Device.getDeviceByMxId('194430106195F41200')]
        # ordered_devs = [dai.Device.getDeviceByMxId('18443010C1A4631200')]
        stream_names = [['b1l', 'b1r', 'b1c', 'b1d']]
        # stream_names = [['box1', 'box2']]
        for dev, sn in zip(device_infos, stream_names):
            time.sleep(1) # Currently required due to XLink race issues
            device: dai.Device = stack.enter_context(
                dai.Device(openvino_version, dev, False))
            # device.setIrLaserDotProjectorBrightness(100)
            device.setIrFloodLightBrightness(50)
            device.startPipeline(create_pipeline(fps, *sn))

            controlQueue = device.getInputQueue('control')

            ctrl = dai.CameraControl()
            ctrl.setFrameSyncMode(dai.RawCameraControl.FrameSyncMode.OUTPUT)
            # ctrl.setStrobeSensor(1)
            controlQueue.send(ctrl)

            logging.warning(device.getOutputQueueNames())
            left_q = device.getOutputQueue(sn[0], maxSize=fps, blocking=True)
            right_q = device.getOutputQueue(sn[1], maxSize=fps, blocking=True)
            color_q = device.getOutputQueue(sn[2], maxSize=fps, blocking=True)
            disparity_q = device.getOutputQueue(sn[3], maxSize=fps, blocking=True)
            left = CameraCapture(left_q, sn[0], fps, width, height, 'h264', 'h264')
            right = CameraCapture(right_q, sn[1], fps, width, height, 'h264', 'h264')
            color = CameraCapture(color_q, sn[2], fps, width, height, 'h264', 'h264')
            disparity = CameraCapture(disparity_q, sn[3], fps, width, height, 'h264', 'h264')
            left.disable_decoding()
            right.disable_decoding()
            color.enable_decoding()
            disparity.enable_decoding()
            left.start_threads()
            right.start_threads()
            color.start_threads()
            disparity.start_threads()
            caps.append([left, right, color, disparity])
            disp_caps.append([color, disparity])

        try:
            images = [[np.zeros((int(height/2), int(width/2), 3))
                       for i in range(len(disp_caps[0]))]
                      for i in range(len(disp_caps))]
            while True:
                changed = False
                for i, cap_row in enumerate(disp_caps):
                    for j, cap in enumerate(cap_row):
                        try:
                            images[i][j] = cap.display_q.get(timeout=0.001)
                            changed = True
                        except queue.Empty:
                            pass
                if changed:
                    # print('Image sizes: {}x{}x{} and {}x{}x{}'.format(*images[0].shape, *images[1].shape))
                    disp_im = np.concatenate([np.concatenate(imrow, axis=1) for imrow in images])
                    cv2.imshow('RodentVision', disp_im)
                    cv2.waitKey(1)
        except KeyboardInterrupt:
            # for cap_row in caps:
            #     for cap in cap_row:
            #         try:
            #             while True:
            #                 cap.display_q.get_nowait()
            #         except queue.Empty:
            #             pass
            cv2.destroyAllWindows()

        for cap_row in caps:
            for cap in cap_row:
                cap.stop_threads()

    logging.info('Exiting...')
