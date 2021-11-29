from collections import defaultdict
from collections import deque

import torch


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def val(self):
        return self.deque[-1]

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return object.__getattr__(self, attr)

    def __str__(self):
        _str = []
        for name, meter in self.meters.items():
            _str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.val, meter.global_avg)
            )
        return self.delimiter.join(_str)

    def tf_write(self, writer, epoch, phase):
        for name, meter in self.meters.items():
            if name not in ['time', 'data']:
                writer.add_scalar('{}/{}'.format(name, phase), meter.avg, epoch)
