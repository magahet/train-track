import pandas as pd
from datetime import (datetime, timedelta)
from collections import deque
import csv


class TimePredictor(object):

    def __init__(self, history_length=5):
        self.history = deque(maxlen=history_length)
        self.stats = None

    def update(self, value):
        self.history.append(value)
        self.calc_stats()

    def calc_stats(self):
        if len(self.history) >= 2:
            history = list(self.history)
            intervals = pd.Series([
                abs((i - j).total_seconds()) for
                (i, j) in
                zip(history, history[1:])
            ])

            self.stats = intervals.describe(90)

    def predict(self):
        '''Makes time predictions based on observed mean and std in seconds.'''
        if self.stats is None or len(self.history) == 0:
            return None
        offset = datetime.now() - self.history[-1]
        print offset
        return {
            'earliest': str(timedelta(seconds=self.stats.loc['5.0%']) - offset),
            'expected': str(timedelta(seconds=self.stats.loc['50%']) - offset),
            'latest': str(timedelta(seconds=self.stats.loc['95%']) - offset),
        }

    def read_log(self, logpath, direction):

        def parse_date(timestamp):
            return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

        with open(logpath, 'r') as log:
            for row in csv.reader(log):
                if row[1] != direction:
                    continue
                self.history.append(parse_date(row[0]))
        self.calc_stats()
