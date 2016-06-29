from nose.tools import eq_
from collections import deque
import datetime
import predict


def test_calc_stats():
    time = datetime.datetime(2000, 01, 01, 00, 00, 00)
    in_ = deque([time])
    eq_(predict.calc_stats(in_), None)
    time += datetime.timedelta(minutes=10)
    in_.append(time)
    eq_(predict.calc_stats(in_), {'mean': 600, 'std': 0})
    time += datetime.timedelta(minutes=20)
    in_.append(time)
    eq_(predict.calc_stats(in_), {'mean': 900, 'std': 300})


def test_make_prediction():
    time = datetime.datetime(2000, 01, 01, 00, 00, 00)
    in_ = {'mean': 600, 'std': 0}
    out = {
        'std': 0,
        'upper99': 600,
        'prediction': datetime.datetime(2000, 1, 1, 0, 10),
        'latest': datetime.datetime(2000, 1, 1, 0, 10),
        'lower99': 600,
        'earliest': datetime.datetime(2000, 1, 1, 0, 10),
        'mean': 600
    }
    eq_(predict.make_prediction(in_, time), out)
    in_ = {'mean': 900, 'std': 300}
    out = {
        'std': 300,
        'upper99': 1800,
        'prediction': datetime.datetime(2000, 1, 1, 0, 15),
        'latest': datetime.datetime(2000, 1, 1, 0, 30),
        'lower99': 0,
        'earliest': datetime.datetime(2000, 1, 1, 0, 0),
        'mean': 900
    }
    eq_(predict.make_prediction(in_, time), out)
