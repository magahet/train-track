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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (21, 21), 0)


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


def draw(frame, thresh, frameDelta, edge_movement, detected):
    # draw the text and timestamp on the frame
    cv2.putText(
        frame, "Edge Movement: {}".format(edge_movement), (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 2)
    cv2.putText(
        frame, "Object Detected: {}".format(detected), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 2)
    cv2.putText(
        frame, datetime.datetime.now(
        ).strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # show the frame and record if the user presses a key
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    # required to display images
    cv2.waitKey(1)


def setup_windows():
    cv2.namedWindow("Security Feed")
    cv2.namedWindow("Thresh")
    cv2.namedWindow("Frame Delta")

    cv2.moveWindow("Security Feed", 0, 0)
    cv2.moveWindow("Thresh", 0, 500)
    cv2.moveWindow("Frame Delta", 0, 1000)


def detect_trains(camera, args):

    # initialize the first frame in the video stream
    detected = False
    moved_out_of_frame = False
    bb_x_ends = deque(maxlen=2)
    movement = deque(maxlen=10)
    edge_movement = 0

    t2 = get_frame()
    t1 = get_frame()
    t0 = get_frame()

    setup_windows()

    # loop over the frames of the video
    while True:

        t2 = t1
        t1 = t0
        t0 = get_frame()

        if t0 is None:
            break

        frame = t0.copy()

        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = diffImg(t0, t1, t2)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        #if args.bounding_box:
            #bx, by, bw, bh =
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)

        # object has left the frame
        if moved_out_of_frame and len(movement) == movement.maxlen:
            print 'Detected Train'
            print np.average(movement)
            movement.clear()

        contour = get_largest_contour(cnts)
        if contour:
            (x, y, w, h) = contour
            bb_x_ends.append((x, x + w))
            edge_movement = (bb_x_ends[-1][0] - bb_x_ends[0][0] +
                             bb_x_ends[-1][1] - bb_x_ends[0][1])
            movement.appendleft(edge_movement)
            # print edge_movement, np.average(movement), max(movement), min(movement)
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            moved_out_of_frame = False
            detected = True
        else:
            moved_out_of_frame = detected
            detected = False

        if not args.console:
            draw(frame, thresh, frameDelta, edge_movement, detected)

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
