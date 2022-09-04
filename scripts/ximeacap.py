from ximea import xiapi
from ximea.xidefs import *
from fractions import Fraction
import cv2
import time
from datetime import datetime
import numpy as np
import threading
import queue
import logging
import sys

# Start CV2 window thread to display images
# MUST BE DONE BEFORE AV IMPORT, SEE:
# https://github.com/PyAV-Org/PyAV/issues/978
# https://github.com/opencv/opencv/issues/21952
if __name__ == '__main__':
    cv2.startWindowThread()
    cv2.namedWindow('RodentVision', cv2.WND_PROP_AUTOSIZE)
    cv2.waitKey(1)

def encode_thread(encq, quit_event, filename, width, height):
    '''Encode images from a queue and save to the provided filename'''
    # setup video context
    import av
    codec = 'hevc_nvenc'
    # codec = 'hevc'
    # codec = 'h264_nvenc'
    avcodec = av.CodecContext.create(codec, 'w')
    # avcodec.time_base = Fraction(1, 1000) # Milliseconds
    output_container = av.open(filename, 'w')
    stream = output_container.add_stream(codec, fps)
    # stream.time_base = Fraction(1, 1000) # Milliseconds
    # t0 = int(time.time() * 1000)
    # stream.start_time = t0
    avcodec.pix_fmt = 'yuv420p'
    stream.width = width
    stream.height = height
    logging.info('Start encoding')
    encode_count = 0
    packet_count = 0
    # time.sleep(0.5)

    while not quit_event.is_set():
        try:
            logging.info('Encoder waiting for data...')
            data = encq.get(timeout=1)
            logging.info('Encoder got data...')
            encode_count += 1
            frame = av.VideoFrame.from_ndarray(data, format='rgb24')
            # frame.pts = int((time.time() - t0) * 1000)
            logging.info('Encoding packet...')
            for packet in stream.encode(frame):
                packet_count += 1
                # packet.pts = int((time.time() - t0) * 1000)
                logging.info('Muxing packet...')
                output_container.mux(packet)
        except queue.Empty:
            pass
    
    logging.info('Clean up encoding')

    # Make sure there are no packets left in the queue
    try:
        logging.info('Encoder looking for remaining data...')
        while True:
            data = encq.get(block=False)
            logging.debug('Encoder got data...')
            encode_count += 1
            frame = av.VideoFrame.from_ndarray(data, format='rgb24')
            # frame.pts = int((time.time() - t0) * 1000)
            logging.debug('Encoding packet...')
            for packet in stream.encode(frame):
                packet_count += 1
                # packet.pts = int((time.time() - t0) * 1000)
                logging.debug('Muxing packet...')
                output_container.mux(packet)
    except queue.Empty:
        pass

    if encode_count > 0:
        for packet in stream.encode():
            packet_count += 1
            output_container.mux(packet)

    logging.info('Encode count: {}\nPacket count: {}'.format(encode_count, packet_count))
    # Close the file
    output_container.close()
    logging.debug('Finished encoding')


def capture_thread(camera, encq, dispq, quit_event):
    '''Capture images from camera and add to the queue'''

    # create instance of Image to store image data and metadata
    logging.info('Capture thread started for camera {}'.format(camera.get_device_sn()))
    img = xiapi.Image()
    capture_count = 0
    camera.set_gpo_selector('XI_GPO_PORT2')
    camera.set_gpo_mode('XI_GPO_EXPOSURE_ACTIVE')
    while not quit_event.is_set():
        # camera.set_param('XI_PRM_TRG_SOFTWARE', 1)
        logging.debug('1')
        camera.set_trigger_software(1)
        logging.debug('2')
        camera.get_image(img)
        data = img.get_image_data_numpy()
        logging.debug('3')
        capture_count += 1
        try:
            # If the encoder is not running, this will cause an indefinite
            # thread lock. I should probably have a warning loop on it.
            encq.put(data)
            logging.debug('4')
        except queue.Full:
            logging.warn('Encoding queue full, dropped frame!')
        try:
            dispq.put(data[::2, ::2].copy(), block=False)
            logging.debug('5')
        except queue.Full:
            pass
    # logging.debug('6')
    camera.set_gpo_selector('XI_GPO_PORT2')
    camera.set_gpo_mode('XI_GPO_OFF')
    logging.info('Capture count for camera {}: {}'.format(camera.get_device_sn(), capture_count))


class CameraCapture():

    def __init__(self, camera_serial, fps, filename, width, height):
        self.camera = xiapi.Camera()
        self.camera_serial = camera_serial
        self.encode_q = queue.Queue(maxsize=fps)
        self.display_q = queue.Queue(maxsize=1)
        self.encoder_quit = threading.Event()
        self.capture_quit = threading.Event()
        self.encoder_thread = threading.Thread(
            target=encode_thread,
            args=(self.encode_q, self.encoder_quit, filename, width, height))
        self.capture_thread = threading.Thread(
            target=capture_thread,
            args=(self.camera, self.encode_q, self.display_q,
                  self.capture_quit))

    def init_camera(self):
        logging.info('Opening camera...')
        self.camera.open_device_by_SN(self.camera_serial)
        logging.info('Opened camera {}...'.format(
            self.camera.get_device_sn()))
        # settings
        self.camera.set_trigger_source('XI_TRG_SOFTWARE')
        self.camera.set_gpo_selector('XI_GPO_PORT2')
        self.camera.set_gpo_mode('XI_GPO_OFF')
        self.camera.set_exposure(35000)
        self.camera.set_gain(10)
        self.camera.set_sensor_bit_depth('XI_BPP_10')
        self.camera.set_output_bit_depth('XI_BPP_8')
        self.camera.set_imgdataformat('XI_RGB24')
        self.camera.set_framerate(fps)
        self.camera.disable_auto_wb()
        self.camera.set_manual_wb(1)

    def start_acquisition(self):
        logging.info('Starting data acquisition for camera {}...'.format(
            self.camera.get_device_sn()))
        self.camera.start_acquisition()
        logging.debug('Started data acquisition for camera {}...'.format(
            self.camera.get_device_sn()))

    def stop_acquisition(self):
        logging.info('Stopping data acquisition for camera {}...'.format(
            self.camera.get_device_sn()))
        self.camera.stop_acquisition()

    def start_encoder(self):
        self.encoder_thread.start()
        while not self.encoder_thread.is_alive():
            time.sleep(0.1)

    def stop_encoder(self):
        logging.info('Stopping encoding queue for camera {}...'.format(
            self.camera.get_device_sn()))
        self.encoder_quit.set()

    def start_capture(self):
        self.capture_thread.start()
        # while not self.capture_thread.is_alive():
        #     time.sleep(0.1)

    def stop_capture(self):
        logging.info('Stopping capture queue for camera {}...'.format(
            self.camera.get_device_sn()))
        self.capture_quit.set()
        while self.capture_thread.is_alive():
            logging.info('Waiting for capture thread to exit...')
            self.capture_thread.join(timeout=0.1)


    def __del__(self):
        self.camera.close_device()


if __name__ == '__main__':
    # codec = 'h264_nvenc'
    log = logging.getLogger()
    log.setLevel(logging.getLevelName('INFO'))
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s [%(threadName)s] ") # I am printing thread id here
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    log.addHandler(console_handler)

    camera_map = ['CACAU2212006', 'CACAU2212013']
    if len(sys.argv) < 3:
        print('Please provide two filenames')
        sys.exit(0)
    if len(sys.argv) > 3:
        print('Please provide two filenames')
        sys.exit(0)
    fns = []
    for arg in sys.argv[1:]:
        fns.append(arg + '.mp4')

    fps = 20
    width = 1936
    height = 1216
    # codec = 'h264'
    t0 = time.time()
    now = datetime.now()
    fn_template = now.strftime('%Y%m%d-%H%M%S-camera-{:02d}.mp4')
    # create instance for all connected cameras
    # ncams = xiapi.Camera().get_number_devices()
    ncams = 2
    caps = [CameraCapture(camera_map[i], fps, fns[i], width, height)
            for i in range(ncams)]
    for cap in caps:
        cap.init_camera()
    for cap in caps:
        cap.start_encoder()
    for cap in caps:
        cap.start_acquisition()
    for cap in caps:
        time.sleep(0.5)
        cap.start_capture()

    try:
        images = [np.zeros((int(height/2), int(width/2), 3)) for i in range(ncams)]
        while True:
            changed = False
            for i, cap in enumerate(caps):
                try:
                    images[i] = cap.display_q.get(timeout=0.005)
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
        cap.stop_capture()
    for cap in caps:
        cap.stop_acquisition()
    for cap in caps:
        cap.stop_encoder()

    logging.info('Exiting...')
