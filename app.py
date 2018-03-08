import argparse
import base64
import logging
from abc import ABC, abstractmethod
from time import sleep
from functools import partial

import cv2
from socketIO_client import SocketIO, BaseNamespace

from pytrackcontrol import TrackEventController
from pytrackcontrol.providers import FPSProvider, FaceBBoxProvider


parser = argparse.ArgumentParser()
parser.add_argument(
    '-d', '--debug',
    help="Debugging mode",
    action="store_const", dest="loglevel", const=logging.DEBUG,
    default=logging.WARNING,
)
parser.add_argument(
    '-v', '--verbose',
    help="Verbose mode",
    action="store_const", dest="loglevel", const=logging.INFO,
)
args = parser.parse_args()
logging.basicConfig(level=args.loglevel)

DEBUG = bool(args.loglevel != logging.WARNING)


class Camera(BaseNamespace):
    pass


socket = SocketIO('127.0.0.1', 5000)
camera_namespace = socket.define(Camera, '/camera')
track = TrackEventController()


# @track.register('img_gray')
# def img_gray_provider(resolve, img):
#     pass  # machine vision
#
#
# @track.register('faces_bbox', dep='img_gray')
# def faces_bbox_finder(resolve, img_gray):
#     pass  # machine vision
#
#
# @track.register('eyes_bbox', dep=['img_gray', 'faces_bbox'])
# def eyes_bbox_finder(resolve, img_gray, faces_bbox):
#     pass  # machine vision
#
#
# @track.register('faces', dep=['img', 'faces_bbox'])
# def faces_finder(resolve, img, faces_bbox):
#     pass  # machine vision
#
#
# @track.on('img')
# def img_handler(img):
#     retval, buffer = cv2.imencode('.jpg', img)
#     encoded_img = str(base64.b64encode(buffer))[2:-1]
#     camera_namespace.emit('publish', {'count': encoded_img}, namespace='/camera')



face_bbox_provider = FaceBBoxProvider()
track.register('faces_bbox', face_bbox_provider.provide)

# separate viola jones from tracker??

@track.register('face', dep=['img', 'faces_bbox'])
def face_provider(resolve, img, faces_bbox):
    if faces_bbox:
        height, width, _ = img.shape
        clamp = lambda minimum, maximum, value: max(minimum, min(maximum, value))
        clamp_x = partial(clamp, 0, width)
        clamp_y = partial(clamp, 0, height)
        x, y, w, h = faces_bbox[0]
        s = int(w * 0.2)
        w = h = w + 2 * s
        x, y = x-s, y-s
        face = img[clamp_y(y) : clamp_y(y+h), clamp_x(x) : clamp_x(x+w)]
        resolve(face)


@track.on('face')
def face_handler(face):
    try:
        retval, buffer = cv2.imencode('.jpg', face)
    except:
        print(face)
    # encoded_img = str(base64.b64encode(buffer))[2:-1]
    encoded_img = base64.b64encode(buffer).decode('UTF-8')
    camera_namespace.emit('publish', {'count': encoded_img}, namespace='/camera')


if DEBUG:
    fps_provider = FPSProvider()
    track.register('fps', fps_provider.provide)

    @track.register('debug', dep=['img', 'faces_bbox', 'fps'])
    def debug_provider(resolve, img, faces_bbox, fps):
        resolve({
            'img': img,
            'faces_bbox': faces_bbox,
            'fps': fps
        })


    @track.on('debug')
    def debug_handler(debug):
        img = debug['img'].copy()
        if debug['faces_bbox']:
            for (x, y, w, h) in debug['faces_bbox']:
                # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0) if jones else (0, 0, 255), 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        img = cv2.flip(img, flipCode=1)
        cv2.putText(img=img, text='{:.2f}'.format(debug['fps']),
                    org=(0, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=150)
        cv2.imshow('img', img)
        cv2.waitKey(10)


track.start()


# track = TrackEventController()


# @track.on('mousemove')
# def mouse_handler(someobj):
#     pass  # invoke some mouse control
#
#
# @track.on('gesture')
# def gesture_handler(someobj):
#     pass  # invoke some gesture control
#
#
# @track.on('face')
# def face_handler(someobj):
#     pass  # invoke some face control
#
#
# @track.on('qr')   # dep=b&w ??? (in mv code)
# def qr_handler(someobj):
#     pass  # invoke some qr control
#
#
# @track.on('img')   # ???
# def img_handler(someobj):
#     pass  # invoke some img control
#
#
# @track.register('cats')
# def cat_finder(resolve, img):   # need state? obj? (might need context manager support if holding resources...)
#     pass  # machine vision
#
#
# @track.register('grumpycats', dep='cats')
# def grumpy_cat_finder(resolve, cat_img):
#     pass  # machine vision


# fps decorator?

# ??? pause the event!!! ..but check for consumers...
# track.pause('cats')   # what about dependencies? =False (default, stop callbacks, but continue eval if other deps)
# track.resume('cats')
#

# add ttl? i.e. get single picture? call pause on the decorator that was saved to dict


#  # server
# socketio = SocketIO(app, message_queue='redis://')
#
#  # CV thread/process?
# socketio = SocketIO(message_queue='redis://')
# socketio.emit('my event', {'data': 'foo'}, namespace='/test')
