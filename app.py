import argparse
import base64
import logging
from functools import partial

import pyautogui
import cv2
import numpy as np
from socketIO_client import SocketIO, BaseNamespace

from pytrackcontrol import TrackEventController
from pytrackcontrol.providers import FPSProvider, FaceBBoxProvider
from pytrackvision.vision.contours import find_contours, find_min_enclosing_circle, find_centroid, find_convex_hull, find_convexity_defects, find_k_curvatures, find_deep_convexity_defects_points, find_max_incircle

from multiprocessing import Process, Pipe


def reader(pipe):
    in_pipe, out_pipe = pipe
    out_pipe.close()    # We are only reading

    width, height = pyautogui.size()
    xx, yy = 0, 0

    while True:
        try:
            x, y = in_pipe.recv()
            y = min(y, 180)

            x = 320 - x
            # x = int((x / 320) * width)
            # x = int(((x / 320) * width) - 0.25 * ((x / 320) * (width/2)))
            x = int(((x / 320) * width) + 0.1 * ((x / 320) * (width/2)))
            # y = int((y / 240) * height)
            # y = int((y / 180) * height)
            y = int(((y / 180) * height) + 0.1 * ((y / 180) * height))

            if abs(xx - x) > 10 or abs(yy - y) > 10:  # TODO
                #pyautogui.moveTo(x, y, duration=0.03, pause=0.0)
                xx, yy = x, y

        except EOFError:
            break


def writer(coor, out_pipe):
    out_pipe.send(coor)


out_pipe, in_pipe = Pipe()
process = Process(target=reader, args=((out_pipe, in_pipe),))
process.start()     # Launch the reader process

out_pipe.close()       # We are only writing


parser = argparse.ArgumentParser()
parser.add_argument(
    '--show-trackers',
    help="Show tracking information",
    action="store_const", dest="show_trackers", const=True,
    default=False,
)
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
SHOW_TRACKERS = args.show_trackers
logging.basicConfig(level=args.loglevel)

DEBUG = bool(args.loglevel != logging.WARNING)


class Camera(BaseNamespace):
    pass


socket = SocketIO('127.0.0.1', 5000)
camera_namespace = socket.define(Camera, '/camera_publish')

track = TrackEventController()


@track.register('img', dep=['src'])
def img_provider(resolve, src):
    # resolve(src[:180])
    resolve(src)


face_bbox_provider = FaceBBoxProvider()
track.register('face_bbox', face_bbox_provider.provide, dep=['img'])

# separate viola jones from tracker??


@track.register('face', dep=['img', 'face_bbox'])
def face_provider(resolve, img, face_bbox):
    if face_bbox:
        height, width, _ = img.shape
        clamp = lambda minimum, maximum, value: max(minimum, min(maximum, value))
        clamp_x = partial(clamp, 0, width)
        clamp_y = partial(clamp, 0, height)
        x, y, w, h = face_bbox
        s = int(w * 0.2)
        w = h = w + 2 * s
        x, y = x-s, y-s
        face = img[clamp_y(y) : clamp_y(y+h), clamp_x(x) : clamp_x(x+w)]
        resolve(face)


# @track.register('face_skin_color_extraction', dep=['img', 'face_bbox'])
# def face_skin_color_extraction_provider(resolve, img, face_bbox):
#     ...


class Range:

    def __init__(self, range):
        """Store a pair of values defining a range with a min and max.
        """
        self.min = range[0]
        self.max = range[1]


class ColorSpaceRange:

    def __init__(self, colors):
        """Store a range for values in a color space.

        Parameters
        ----------
        colors: Dict[str: Tuple[int]]
            A map of color space labels and a range of values.
        """
        self._colors = {k: Range(v) for k, v in colors.items()}

    def __getitem__(self, key):
        return self._colors[key]

    @property
    def colors(self):
        """
        Returns
        -------
        Dict[str: Range]
        """
        return self._colors


def create_color_range_slider(title, color_space_range):
    """Create a slider to adjust the color space range.

    Parameters
    ----------
    title: str
    color_space_range: ColorSpaceRange
    """

    def _update(color_range, attr):
        def _set(v):
            if attr == 'min':
                color_range.min = v
            elif attr == 'max':
                color_range.max = v

        return _set

    cv2.namedWindow(title)
    for label, color_range in color_space_range.colors.items():
        cv2.createTrackbar(f'{label} min', title, color_range.min, 255, _update(color_range, 'min'))
        cv2.createTrackbar(f'{label} max', title, color_range.max, 255, _update(color_range, 'max'))


# # Lab black screen
# yCrCb_range = ColorSpaceRange({
#     'y':  (131, 255),
#     'cr': (129, 176),
#     'cb': (0,   255),
# })
#
# HSV_range = ColorSpaceRange({
#     'h': (106, 255),
#     's': (12,  70),
#     'v': (150, 247),
# })

# Lab wall
yCrCb_range = ColorSpaceRange({
    'y':  (118, 255),
    'cr': (133, 159),
    'cb': (104, 136),
})

HSV_range = ColorSpaceRange({
    'h': (0,   22),
    's': (34,  105),
    'v': (141, 215),
})

# # Home
# yCrCb_range = ColorSpaceRange({
#     'y':  (144, 255),
#     'cr': (0,   255),
#     'cb': (0,   255),
# })
#
# HSV_range = ColorSpaceRange({
#     'h': (0,   255),
#     's': (0,   39),
#     'v': (146, 247),
# })
#


@track.register('img_ycrcb', dep=['img'])
def ycrcb_color_conversion_provider(resolve, img):
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # img_ycrcb = cv2.GaussianBlur(img_ycrcb, (3, 3), 0)
    # img_ycrcb = cv2.medianBlur(img_ycrcb, ksize=5)
    resolve(img_ycrcb)


@track.register('img_hsv', dep=['img'])
def hsv_color_conversion_provider(resolve, img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # img_hsv = cv2.GaussianBlur(img_hsv, (3, 3), 0)
    # img_hsv = cv2.medianBlur(img_hsv, ksize=5)
    resolve(img_hsv)


# def get_bounds(roi):
#     roi = roi.copy()
#     size = len(roi)
#     pts = [
#                     (1/3, 1/6), (1/2, 1/6), (2/3, 1/6),
#         (3/12, 3/6), (1/3, 3/6), (1/2, 3/6), (2/3, 3/6), (9/12, 3/6),
#                     (1/3, 4/6), (1/2, 4/6), (2/3, 4/6),
#     ]
#     w = 5
#     h = 5
#     pt_samples = []
#     for x, y in pts:
#         x = int(size * x)
#         y = int(size * y)
#         pt_roi = roi[y: y + h, x: x + w]
#         average_color = [np.median(pt_roi[:, :, i]) for i in range(pt_roi.shape[-1])]
#         pt_samples.append(average_color)
#         cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 1)
#
#     lower = np.min(pt_samples, axis=0)
#     upper = np.max(pt_samples, axis=0)
#     # lower = np.min(pt_samples, axis=0)*.75 + np.median(pt_samples, axis=0)*.25
#     # upper = np.max(pt_samples, axis=0)*.75 + np.median(pt_samples, axis=0)*.25
#
#     cv2.imshow('pic', roi)
#     return lower, upper
#
#
# @track.register('ycrcb_range', dep=['img_ycrcb', 'face_bbox'])
# def ycrcb_color_range_provider(resolve, img, bbox):
#     x, y, w, h = bbox
#     roi = img[y: y + h, x: x + w]
#     color_range_arr = get_bounds(roi)
#     color_range = ColorSpaceRange({
#         'y':  (color_range_arr[0][0], color_range_arr[1][0]),
#         'cr': (color_range_arr[0][1], color_range_arr[1][1]),
#         'cb': (color_range_arr[0][2], color_range_arr[1][2]),
#     })
#     resolve(color_range)
#
#
# @track.register('hsv_range', dep=['img_hsv', 'face_bbox'])
# def hsv_color_range_provider(resolve, img, bbox):
#     x, y, w, h = bbox
#     roi = img[y: y + h, x: x + w]
#     color_range_arr = get_bounds(roi)
#     color_range = ColorSpaceRange({
#         'h': (color_range_arr[0][0], color_range_arr[1][0]),
#         's': (color_range_arr[0][1], color_range_arr[1][1]),
#         'v': (color_range_arr[0][2], color_range_arr[1][2]),
#     })
#     resolve(color_range)


@track.register('ycrcb_range', dep=['img'])
def ycrcb_color_range_provider(resolve, img):
    resolve(yCrCb_range)


@track.register('hsv_range', dep=['img'])
def hsv_color_range_provider(resolve, img):
    resolve(HSV_range)


@track.register('ycrcb_mask', dep=['img_ycrcb', 'ycrcb_range'])
def ycrcb_color_threshold_mask_provider(resolve, img_ycrcb, ycrcb_range):
    img_y, img_cr, img_cb = cv2.split(img_ycrcb)

    mask_y = cv2.inRange(img_y, ycrcb_range['y'].min, ycrcb_range['y'].max)
    mask_cr = cv2.inRange(img_cr, ycrcb_range['cr'].min, ycrcb_range['cr'].max)
    mask_cb = cv2.inRange(img_cb, ycrcb_range['cb'].min, ycrcb_range['cb'].max)

    skin_mask_ycrcb = cv2.bitwise_and(mask_cb, mask_cr)
    skin_mask_ycrcb = cv2.bitwise_and(mask_y, skin_mask_ycrcb)
    resolve(skin_mask_ycrcb)

    if DEBUG:
        cv2.imshow('ycrcb_mask', np.vstack((mask_y, mask_cr, mask_cb, skin_mask_ycrcb)))


@track.register('hsv_mask', dep=['img_hsv', 'hsv_range'])
def hsv_color_threshold_mask_provider(resolve, img_hsv, hsv_range):
    img_h, img_s, img_v = cv2.split(img_hsv)

    mask_h = cv2.inRange(img_h, hsv_range['h'].min, hsv_range['h'].max)
    mask_s = cv2.inRange(img_s, hsv_range['s'].min, hsv_range['s'].max)
    mask_v = cv2.inRange(img_v, hsv_range['v'].min, hsv_range['v'].max)

    skin_mask_hsv = cv2.bitwise_and(mask_v, mask_s)
    skin_mask_hsv = cv2.bitwise_and(mask_h, skin_mask_hsv)
    resolve(skin_mask_hsv)

    if DEBUG:
        cv2.imshow('hsv_mask', np.vstack((mask_h, mask_s, mask_v, skin_mask_hsv)))


@track.register('skin_segmentation_mask', dep=['ycrcb_mask', 'hsv_mask', 'face_bbox'])
def skin_segmentation_mask_provider(resolve, ycrcb_mask, hsv_mask, bbox):
    skin_mask = cv2.bitwise_or(ycrcb_mask, hsv_mask)

    x, y, w, h = bbox
    iw, ih = ycrcb_mask.shape
    skin_mask[y - int(.2 * h): y + h + int(.4 * h), x: x + w] = 0

    resolve(skin_mask)

# def a(ximg_ycrcb, xycrcb_range):
#     import cv2
#     img_y, img_cr, img_cb = cv2.split(ximg_ycrcb)
#
#     mask_y = cv2.inRange(img_y, xycrcb_range['y'].min, xycrcb_range['y'].max)
#     mask_cr = cv2.inRange(img_cr, xycrcb_range['cr'].min, xycrcb_range['cr'].max)
#     mask_cb = cv2.inRange(img_cb, xycrcb_range['cb'].min, xycrcb_range['cb'].max)
#
#     ycrcb_mask = cv2.bitwise_and(mask_cb, mask_cr)
#     ycrcb_mask = cv2.bitwise_and(mask_y, ycrcb_mask)
#     return ycrcb_mask
#
# def b(ximg_hsv, xhsv_range):
#     import cv2
#     img_h, img_s, img_v = cv2.split(ximg_hsv)
#
#     mask_h = cv2.inRange(img_h, xhsv_range['h'].min, xhsv_range['h'].max)
#     mask_s = cv2.inRange(img_s, xhsv_range['s'].min, xhsv_range['s'].max)
#     mask_v = cv2.inRange(img_v, xhsv_range['v'].min, xhsv_range['v'].max)
#
#     hsv_mask = cv2.bitwise_and(mask_v, mask_s)
#     hsv_mask = cv2.bitwise_and(mask_h, hsv_mask)
#
# @track.register('skin_segmentation_mask', dep=['img_ycrcb', 'ycrcb_range', 'img_hsv', 'hsv_range', 'face_bbox'])
# def skin_segmentation_mask_provider(resolve, img_ycrcb, ycrcb_range, img_hsv, hsv_range, bbox):
#
#
#     # futures = [self._executor.submit(execute, fn, e, inputs) for e, (fn, res, inputs) in jobs.items()]
#     #                     for future in futures:
#     #                         res(future.result())
#     fa = executor.submit(a, img_ycrcb, ycrcb_range)
#     fb = executor.submit(b, img_hsv, hsv_range)
#
#     ycrcb_mask = fa.result()
#     hsv_mask = fb.result()
#     print(ycrcb_mask)
#     skin_mask = cv2.bitwise_or(ycrcb_mask, hsv_mask)
#
#     x, y, w, h = bbox
#     iw, ih = ycrcb_mask.shape
#     skin_mask[y - int(.2 * h): y + h + int(.4 * h), x: x + w] = 0
#
#     resolve(skin_mask)


@track.register('skin_mask', dep=['skin_segmentation_mask'])
def skin_mask_provider(resolve, skin_segmentation_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    skin_mask = cv2.morphologyEx(skin_segmentation_mask, cv2.MORPH_ERODE, kernel)
    cv2.morphologyEx(skin_mask, cv2.MORPH_DILATE, kernel, dst=skin_mask)

    cv2.medianBlur(skin_mask, ksize=7, dst=skin_mask)

    resolve(skin_mask)


@track.register('contours', dep=['skin_mask'])
def contours_provider(resolve, skin_mask):
    contours = find_contours(skin_mask, min_area=800, max_area=8000)
    resolve(contours)


@track.register('contours_min_enclosing_circle', dep=['contours'])
def contours_min_enclosing_circle_provider(resolve, contours):
    circles = []
    for cnt in contours:
        center, radius = find_min_enclosing_circle(cnt)
        circles.append((center, radius))

    resolve(circles)


@track.register('contours_moments_centroid', dep=['contours'])
def contours_moments_centroid_provider(resolve, contours):
    centroids = []
    for cnt in contours:
        cX, cY = find_centroid(cnt)
        centroids.append((cX, cY))

    resolve(centroids)


@track.register('contours_convex_hulls', dep=['contours'])
def contours_convex_hulls_provider(resolve, contours):
    convex_hulls = []
    for cnt in contours:
        hull = find_convex_hull(cnt)
        convex_hulls.append(hull)

    resolve(convex_hulls)


@track.register('k_curvatures', dep=['contours', 'contours_convex_hulls'])
def k_curvatures_provider(resolve, contours, convex_hulls):
    k_curvatures = []
    for cnt, hull in zip(contours, convex_hulls):
        curvatures = find_k_curvatures(cnt, hull, k=7, theta=60)
        k_curvatures.append(curvatures)

    resolve(k_curvatures)


@track.register('contours_convexity_defects', dep=['contours', 'contours_convex_hulls'])
def contours_convexity_defects_provider(resolve, contours, convex_hulls):
    convexity_defects = []
    for cnt, hull in zip(contours, convex_hulls):
        defects = find_convexity_defects(cnt)
        convexity_defects.append(defects)

    resolve(convexity_defects)


@track.register('contours_deep_convexity_defects_points', dep=['contours', 'contours_convexity_defects'])
def contours_deep_convexity_defects_points_provider(resolve, contours, convexity_defects):
    # import pdb; pdb.set_trace()
    deep_convexity_defects_points = []
    for cnt, defects in zip(contours, convexity_defects):
        points = find_deep_convexity_defects_points(cnt, defects)
        deep_convexity_defects_points.append(points)
    resolve(deep_convexity_defects_points)


def nearest_point_distance(node, nodes):
    deltas = nodes - node
    dist = np.linalg.norm(deltas, axis=1)
    # min_idx = np.argmin(dist)
    # return nodes[min_idx], dist[min_idx], deltas[min_idx][1]/deltas[min_idx][0]
    return np.min(dist)


@track.register('contours_max_incircle', dep=['contours', 'contours_moments_centroid', 'contours_deep_convexity_defects_points'])
def contours_max_incircle_provider(resolve, contours, centroids, deep_convexity_defects_points):
    incircles = []
    for cnt, centroid, defects_points in zip(contours, centroids, deep_convexity_defects_points):
        points = cnt[::5][:, 0]  # Sample the contour
        points = np.concatenate((points, defects_points), axis=0) if defects_points.size else points
        incircle = find_max_incircle(centroid, points)
        incircles.append(incircle)

    resolve(incircles)


@track.register('hands', dep=['contours_moments_centroid'])
def hands_provider(resolve, centroids):
    resolve(centroids)


@track.on('hands')
def hands_handler(hands):
    # width, height = pyautogui.size()
    for (x, y) in hands:
        # x = 320 - x
        # # x = int((x / 320) * width)
        # # x = int(((x / 320) * width) - 0.25 * ((x / 320) * (width/2)))
        # x = int(((x / 320) * width) + 0.1 * ((x / 320) * (width/2)))
        # # y = int((y / 240) * height)
        # # y = int((y / 180) * height)
        # y = int(((y / 180) * height) + 0.1 * ((y / 180) * height))
        # pyautogui.moveTo(x, y, duration=0.0)
        writer((x, y), in_pipe)


@track.on('face')
def face_handler(face):
    try:
        retval, buffer = cv2.imencode('.jpg', face)
    except Exception:
        print(face)
    encoded_img = base64.b64encode(buffer).decode('UTF-8')
    camera_namespace.emit('publish', {'count': encoded_img}, namespace='/camera_publish')
    socket.wait(0.001)


if DEBUG:
    create_color_range_slider('yCrCb Color Settings', yCrCb_range)
    create_color_range_slider('HSV Color Settings', HSV_range)

    @track.on('img_ycrcb')
    def ycrcb_color_conversion_debug_handler(ycrcb_img):
        cv2.imshow('img_ycrcb', ycrcb_img)

    @track.on('img_hsv')
    def hsv_color_conversion_debug_handler(hsv_img):
        cv2.imshow('img_hsv', hsv_img)

    @track.on('skin_segmentation_mask')
    def skin_segmentation_mask_debug_handler(skin_segmentation_mask):
        cv2.imshow('skin_segmentation_mask', skin_segmentation_mask)

    @track.on('skin_mask')
    def skin_mask_debug_handler(skin_mask):
        cv2.imshow('skin_mask', skin_mask)

if SHOW_TRACKERS or DEBUG:
    fps_provider = FPSProvider()
    track.register('fps', fps_provider.provide)

    @track.register('debug', dep=['img', 'fps', 'face_bbox', 'contours', 'contours_min_enclosing_circle', 'contours_moments_centroid', 'contours_convex_hulls', 'k_curvatures', 'contours_convexity_defects', 'contours_deep_convexity_defects_points', 'contours_max_incircle'])
    def debug_provider(resolve, img, fps, face_bbox, contours, min_enclosing_circles, centroids, convex_hulls, k_curvatures, convexity_defects, deep_convexity_defects_points, max_incircles):
        resolve({
            'img': img,
            'fps': fps,
            'face_bbox': face_bbox,
            'contours': contours,
            'min_enclosing_circles': min_enclosing_circles,
            'centroids': centroids,
            'convex_hulls': convex_hulls,
            'k_curvatures': k_curvatures,
            'convexity_defects': convexity_defects,
            'deep_convexity_defects_points': deep_convexity_defects_points,
            'max_incircles': max_incircles
        })

    @track.on('debug')
    def debug_handler(debug):
        img = debug['img']
        contours = debug['contours']
        min_enclosing_circles = debug['min_enclosing_circles']
        centroids = debug['centroids']
        convex_hulls = debug['convex_hulls']
        k_curvatures = debug['k_curvatures']
        convexity_defects = debug['convexity_defects']
        deep_convexity_defects_points = debug['deep_convexity_defects_points']
        max_incircles = debug['max_incircles']

        # flooded_contours = np.zeros((height, width), np.uint8)

        for cnt, enclosing_circle, (cX, cY), hull, k_curvature, defects, defects_deep, incircle in zip(contours, min_enclosing_circles, centroids, convex_hulls, k_curvatures, convexity_defects, deep_convexity_defects_points, max_incircles):
            cv2.drawContours(img, [cnt], 0, (0, 255, 0), cv2.FILLED)
            # cv2.drawContours(flooded_contours, [cnt], 0, 255, cv2.FILLED)

            (center, radius) = enclosing_circle
            cv2.circle(img, center, radius, (0, 0, 0), 1)

            cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)

            for i in k_curvature:
                point = cnt[i][0][0]
                cv2.circle(img, (point[0], point[1]), 4, [255, 0, 0], 2)
                point = cnt[(i + 7) % cnt.shape[0]][0][0]
                cv2.circle(img, (point[0], point[1]), 1, [255, 255, 255], -1)
                point = cnt[i - 7][0][0]
                cv2.circle(img, (point[0], point[1]), 1, [255, 255, 255], -1)

            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                cv2.line(img, start, end, [0, 255, 255], 1)

            for far in defects_deep:
                cv2.circle(img, tuple(far), 2, [0, 0, 255], -1)

            if incircle:
                (center, radius) = incircle
                cv2.circle(img, center, radius, [0, 0, 0], 1)
            else:
                print('incircle not found')

        if debug['face_bbox']:
            (x, y, w, h) = debug['face_bbox']
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0) if jones else (0, 0, 255), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # img = cv2.flip(img, flipCode=1)
        cv2.putText(img=img, text='{:.2f}'.format(debug['fps']),
                    org=(0, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=255)

        # cv2.rectangle(img, (0, 0), (320, 180), (0, 0, 255), 3)
        # cv2.imshow('img', img[:, :, :180])
        cv2.imshow('img', img)
        cv2.waitKey(10)


track.start()


# ??? pause the event!!! ..but check for consumers...
# track.pause('cats')   # what about dependencies? =False (default, stop callbacks, but continue eval if other deps)
# track.resume('cats')
#

# add ttl? i.e. get single picture? call pause on the decorator that was saved to dict
