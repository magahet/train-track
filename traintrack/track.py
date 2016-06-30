import datetime
import time
import numpy as np
from collections import deque
import camera
import cv2
import logging
import os


logging.getLogger(__name__).addHandler(logging.NullHandler())


class TrainTracker(object):
    '''Tracks the passing of trains and predicts next arrival.'''

    def __init__(self, source, video=False, slow=False, bb_str=None, logpath='/tmp/train-track.log', imgdir='/tmp/train-track/'):
        self.diff_thresh = 10
        self.video = video
        self.slow = slow
        self.bb = None
        if bb_str:
            bb = [int(i) for i in bb_str.split('x')]
            if len(bb) == 4:
                self.bb = bb
        self.cam = camera.Camera(source, self.bb)
        self.logpath = logpath
        self.imgdir = imgdir

    def __del__(self):
        logging.info('Stopping Tracker')

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

        logging.info('Starting Tracker')
        section_diffs = deque(maxlen=6)
        #predictors = {
            #'east': predict.TimePredictor(5),
            #'west': predict.TimePredictor(5),
        #}
        cooldown = 0
        last_dict = {
            'east': None,
            'west': None,
        }
        for frame, diff, thresh in self.cam.iter_motion():
            pdiff = calc_section_diff(thresh)
            if abs(pdiff) > self.diff_thresh:
                section_diffs.append(pdiff)
            else:
                cooldown = max(cooldown - 1, 0)
                section_diffs.clear()
            event_detected = (
                len(section_diffs) == section_diffs.maxlen and
                abs(np.average(section_diffs)) > self.diff_thresh * 5 and
                not cooldown
            )
            if event_detected:
                cooldown = 10
                direction = 'west' if sum(section_diffs) > 0 else 'east'
                now = datetime.datetime.now()
                last = last_dict.get(direction)
                self.log(now, direction)
                if last is not None:
                    interval = now - last
                    logging.info(
                        "Section diff sum: [%d], Direction: [%s], Interval: [%d]",
                        sum(section_diffs), direction, interval.total_seconds())
                else:
                    logging.info(
                        "Section diff sum: [%d], Direction: [%s], Interval: []",
                        sum(section_diffs), direction)
                #predictor = predictors.get(direction)
                #predictor.update(now)
                #print predictor.get_prediction()
                section_diffs.clear()
                last_dict[direction] = now

            if self.video:
                self.draw([frame, diff, thresh], self.bb)

            if self.slow:
                time.sleep(0.25)

    def log(self, timestamp, direction):
        with open(self.logpath, 'a') as log:
            log.write('{:%Y-%m-%d %H:%M:%S},{}\n'.format(timestamp, direction))
        if not os.path.isdir(self.imgdir):
            os.makedirs(self.imgdir)
        filepath = os.path.join(self.imgdir, '{:%Y-%m-%d %H:%M:%S}.jpg'.format(timestamp))
        self.cam.save_last_frame(filepath)
