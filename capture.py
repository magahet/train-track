#!/usr/bin/env python


# import the necessary packages
import argparse
import datetime
import imutils
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


def get_frame():
    # resize the frame, convert it to grayscale, and blur it
    grabbed, frame = camera.read()
    if not grabbed:
        return None
    frame = imutils.resize(frame, width=500)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #return cv2.GaussianBlur(gray, (21, 21), 0)


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


def draw(frame, rgb, bb, flow):
    # draw the text and timestamp on the frame
    #cv2.putText(
        #frame, "Edge Movement: {}".format(edge_movement), (10, 40),
        #cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 2)
    #cv2.putText(
        #frame, "Object Detected: {}".format(detected), (10, 20),
        #cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 2)
    x, y, w, h = bb
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.putText(
        frame, datetime.datetime.now(
        ).strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # show the frame and record if the user presses a key
    cv2.imshow('Feed', frame)
    cv2.imshow('Motion', rgb)
    cv2.imshow('Flow', flow)
    # required to display images
    cv2.waitKey(1)


def draw_flow(frame, flow, step=16):
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
    cv2.namedWindow("Feed")
    cv2.namedWindow("Motion")
    cv2.namedWindow("Flow")
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
    last_time = time.time() - 60
    threshold = 2

    setup_windows()

    magnatude = deque(maxlen=10)

    bb = 520, 290, 200, 30
    x, y, w, h = bb

    ret, frame1 = cap.read()
    roi = frame1[y:y + h, x:x + w]
    prvs = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(roi)
    hsv[..., 1] = 255

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        roi = frame2[y:y + h, x:x + w]
        next_ = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prvs, next_, 0.5, 1, 5, 15, 3, 5, 1)

        #flow = cv2.calcOpticalFlowFarneback(
            #prvs,
            #next_,
            #None,
            #0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        mag[np.abs(mag) < threshold] = 0
        mag_sum = mag.sum()
        if len(magnatude) == magnatude.maxlen and time.time() - last_time > dead_time:
            if mag_sum > np.mean(magnatude) + 100 * np.std(magnatude):
                ang_mean = np.average(ang, weights=mag)
                direction = 'west' if (
                    ang_mean < 90 or ang_mean > 270) else 'east'
                print mag_sum, ang_mean, direction, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")
                last_time = time.time()
        magnatude.append(mag_sum)
        #hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 0] = ang
        #print hsv[..., ]
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        prvs = next_

        if not args.console:
            draw(frame2, rgb, bb, draw_flow(roi, flow))

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
    detect_trains(camera, args)
