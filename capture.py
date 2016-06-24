#!/usr/bin/env python


# import the necessary packages
import argparse
import datetime
#import imutils
import time
import cv2
import sys
import signal
import os
import numpy as np
from collections import deque


def signal_handler(signal, frame):
    print 'Exiting'
    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
    sys.exit(0)


def diffImg(t0, t1, t2):
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)


def get_frame(bb=None):
    # resize the frame, convert it to grayscale, and blur it
    grabbed, frame = camera.read()
    if not grabbed:
        return None, None
    if bb:
        x, y, w, h = bb
        roi = frame[y:y + h, x:x + w]
    else:
        roi = frame
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return frame, gray


def get_largest_contour(cnts):
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < args.min_area:
        return None


def get_video(args):
    # if the video argument is None, then we are reading from webcam
    if not args.video:
        camera = cv2.VideoCapture(1)
        time.sleep(0.5)
        while not camera.isOpened():
            time.sleep(0.25)
        print 'adjusting to the light'
        for i in xrange(100):
            camera.read()
        print 'ready'
        return camera

    # otherwise, we are reading from a video file
    else:
        return cv2.VideoCapture(os.path.abspath(args.video))


def draw(frames, bb=None):
    # draw the text and timestamp on the frame
    #cv2.putText(
        #frame, "Edge Movement: {}".format(edge_movement), (10, 40),
        #cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 2)
    #cv2.putText(
        #frame, "Object Detected: {}".format(detected), (10, 20),
        #cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 2)
    if not frames:
        return
    if bb:
        x, y, w, h = bb
        cv2.rectangle(frames[0], (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.putText(
        frames[0], datetime.datetime.now(
        ).strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frames[0].shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    offset = 0
    for num, frame in enumerate(frames):
        cv2.imshow('Frame {}'.format(num), frame)
        cv2.moveWindow('Feed {}'.format(num), 0, offset)
        offset += frame.shape[1]

    # required to display images
    cv2.waitKey(1)


def draw_flow(frame, flow, step=8):
    h, w = frame.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).reshape(-1, 2, 2)
    lines = np.int32(lines)

    vis = frame.copy()
    for (x1, y1), (x2, y2) in lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def setup_windows():
    cv2.namedWindow("Feed", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Motion", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Flow", cv2.WINDOW_NORMAL)
    #cv2.namedWindow("Security Feed")
    #cv2.namedWindow("Thresh")
    #cv2.namedWindow("Frame Delta")

    cv2.moveWindow("Feed", 0, 0)
    cv2.moveWindow("Motion", 0, 500)
    cv2.moveWindow("Flow", 0, 600)
    #cv2.moveWindow("Security Feed", 0, 0)
    #cv2.moveWindow("Thresh", 0, 500)
    #cv2.moveWindow("Frame Delta", 0, 1000)


def detect_trains(cap, args):

    dead_time = 60
    last_event = time.time() - 60
    diff_thresh = 10

    #setup_windows()

    #magnatude = deque(maxlen=10)

    bb = 520, 250, 200, 70

    _, t_minus = get_frame(bb)
    _, t = get_frame(bb)
    _, t_plus = get_frame(bb)

    d = deque(maxlen=6)
    while True:
        t_minus = t
        t = t_plus
        frame, t_plus = get_frame(bb)
        if frame is None:
            break

        diff = diffImg(t_minus, t, t_plus)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        if time.time() - last_event > dead_time:
            if len(d) < d.maxlen:
                l, r = np.hsplit(thresh, 2)
                pdiff = np.count_nonzero(l) - np.count_nonzero(r)
                if abs(pdiff) > diff_thresh:
                    d.append(pdiff)
                else:
                    d.clear()
            else:
                last_event = time.time()
                direction = 'west' if sum(d) > 0 else 'east'
                print sum(d), d, direction
                d.clear()

        if not args.console:
            draw([frame, diff, thresh], bb)

        if args.slow:
            time.sleep(0.25)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument(
        "-a", "--min-area", type=int, default=100, help="minimum area size")
    ap.add_argument(
        "-c", "--console", default=False, help="run in console mode", action='store_true')
    ap.add_argument(
        "-s", "--slow", default=False, help="run in slow mode", action='store_true')
    args = ap.parse_args()

    camera = get_video(args)
    result = detect_trains(camera, args)
