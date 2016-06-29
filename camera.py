import cv2
import time
from collections import deque


class Camera(object):
    '''Represents the camera interface for a connected video device.'''

    def __init__(self, source, bb=None):
        source = int(source) if source.isdigit() else source
        self.camera = cv2.VideoCapture(source)
        self.bb = bb
        self.frames = deque(maxlen=3)
        if isinstance(source, int):
            while not self.camera.isOpened():
                time.sleep(0.25)
            print 'adjusting to the light'
            for _, _, thresh in self.iter_motion():
                if thresh.sum() == 0:
                    break
            print 'ready'

    def __del__(self):
        self.camera.release()

    def get_frame(self):
        # resize the frame, convert it to grayscale, and blur it
        grabbed, frame = self.camera.read()
        if not grabbed:
            return None, None
        if self.bb:
            x, y, w, h = self.bb
            roi = frame[y:y + h, x:x + w]
        else:
            roi = frame
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        return frame, gray

    def get_delta(self):
        if len(self.frames) < 3:
            return None
        return cv2.bitwise_and(
            cv2.absdiff(self.frames[2], self.frames[1]),
            cv2.absdiff(self.frames[1], self.frames[0])
        )

    def iter_motion(self):
        while self.camera.isOpened():
            frame, gray = self.get_frame()
            if frame is not None:
                self.frames.append(gray)
            delta = self.get_delta()
            if delta is None:
                continue
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            yield frame, delta, thresh
