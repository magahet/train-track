import numpy as np
import datetime
from collections import deque


class TimePredictor(object):

    def __init__(self, history_length=5):
        self.history = deque(maxlen=history_length)
        self.mean = None
        self.std = None

    def update(self, value):
        self.history.append(value)
        if len(self.history) >= 2:
            history = list(self.history)
            intervals = [
                abs((i - j).total_seconds()) for
                (i, j) in
                zip(history, history[1:])
            ]
            self.mean = np.average(intervals)
            self.std = np.std(intervals)

    def get_prediction(self):
        '''Makes time predictions based on observed mean and std in seconds.'''
        if None in (self.mean, self.std):
            return None
        now = datetime.datetime.now()
        lower99 = self.mean - 3 * self.std
        upper99 = self.mean + 3 * self.std
        return {
            'mean': self.mean,
            'std': self.std,
            'lower99': lower99,
            'upper99': upper99,
            'prediction': now + datetime.timedelta(seconds=self.mean),
            'earliest': now + datetime.timedelta(seconds=lower99),
            'latest': now + datetime.timedelta(seconds=upper99),
        }
