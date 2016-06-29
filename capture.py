#!/usr/bin/env python


import argparse
import datetime
import time
import numpy as np
from collections import deque
import predict
import camera
import cv2


class TrainTracker(object):
    '''Tracks the passing of trains and predicts next arrival.'''

    def __init__(self, source, video=False, slow=False, bb_str=None):
        self.diff_thresh = 10
        self.video = video
        self.slow = slow
        self.bb = None
        if bb_str:
            bb = [int(i) for i in bb_str.split('x')]
            if len(bb) == 4:
                self.bb = bb
        self.cam = camera.Camera(source, self.bb)

    @staticmethod
    def draw(frames, bb=None):
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

    def run(self):

        def calc_section_diff(array):
            l, r = np.hsplit(array, 2)
            return np.count_nonzero(l) - np.count_nonzero(r)

        section_diffs = deque(maxlen=6)
        predictors = {
            'east': predict.TimePredictor(5),
            'west': predict.TimePredictor(5),
        }
        cooldown = 0
        for frame, diff, thresh in self.cam.iter_motion():
            pdiff = calc_section_diff(thresh)
            if abs(pdiff) > self.diff_thresh:
                section_diffs.append(pdiff)
            else:
                cooldown = max(cooldown - 1, 0)
                section_diffs.clear()
            if len(section_diffs) == section_diffs.maxlen and not cooldown:
                cooldown = 10
                direction = 'west' if sum(section_diffs) > 0 else 'east'
                now = datetime.datetime.now()
                print now.strftime("%A %d %B %Y %I:%M:%S%p"), sum(section_diffs), section_diffs, direction
                predictor = predictors.get(direction)
                predictor.update(now)
                print predictor.get_prediction()
                section_diffs.clear()

            if self.video:
                self.draw([frame, diff, thresh], self.bb)

            if self.slow:
                time.sleep(0.25)


if __name__ == '__main__':
    #signal.signal(signal.SIGINT, signal_handler)
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", dest='input_', help="path to the video source or device number", default='0')
    ap.add_argument(
        "-v", "--video", default=False, help="Show capture and motion images", action='store_true')
    ap.add_argument(
        "-s", "--slow", default=False, help="run in slow mode", action='store_true')
    ap.add_argument("-b", "--bb", help="Optional bounding box")
    args = ap.parse_args()

    tracker = TrainTracker(args.input_, args.video, args.slow, args.bb)
    #self.bb = 520, 250, 200, 70
    try:
        tracker.run()
    except KeyboardInterrupt:
        print 'Stopping'
        del tracker
