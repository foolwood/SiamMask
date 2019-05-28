# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import numpy as np


class Meter(object):
    def __init__(self, name, val, avg):
        self.name = name
        self.val = val
        self.avg = avg

    def __repr__(self):
        return "{name}: {val:.6f} ({avg:.6f})".format(
            name=self.name, val=self.val, avg=self.avg
        )

    def __format__(self, *tuples, **kwargs):
        return self.__repr__()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = {}
        self.sum = {}
        self.count = {}

    def update(self, batch=1, **kwargs):
        val = {}
        for k in kwargs:
            val[k] = kwargs[k] / float(batch)
        self.val.update(val)
        for k in kwargs:
            if k not in self.sum:
                self.sum[k] = 0
                self.count[k] = 0
            self.sum[k] += kwargs[k]
            self.count[k] += batch

    def __repr__(self):
        s = ''
        for k in self.sum:
            s += self.format_str(k)
        return s

    def format_str(self, attr):
        return "{name}: {val:.6f} ({avg:.6f}) ".format(
                    name=attr,
                    val=float(self.val[attr]),
                    avg=float(self.sum[attr]) / self.count[attr])

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return super(AverageMeter, self).__getattr__(attr)
        if attr not in self.sum:
            # logger.warn("invalid key '{}'".format(attr))
            print("invalid key '{}'".format(attr))
            return Meter(attr, 0, 0)
        return Meter(attr, self.val[attr], self.avg(attr))

    def avg(self, attr):
        return float(self.sum[attr]) / self.count[attr]


class IouMeter(object):
    def __init__(self, thrs, sz):
        self.sz = sz
        self.iou = np.zeros((sz, len(thrs)), dtype=np.float32)
        self.thrs = thrs
        self.reset()

    def reset(self):
        self.iou.fill(0.)
        self.n = 0

    def add(self, output, target):
        if self.n >= len(self.iou):
            return
        target, output = target.squeeze(), output.squeeze()
        for i, thr in enumerate(self.thrs):
            pred = output > thr
            mask_sum = (pred == 1).astype(np.uint8) + (target > 0).astype(np.uint8)
            intxn = np.sum(mask_sum == 2)
            union = np.sum(mask_sum > 0)
            if union > 0:
                self.iou[self.n, i] = intxn / union
            elif union == 0 and intxn == 0:
                self.iou[self.n, i] = 1
        self.n += 1

    def value(self, s):
        nb = max(int(np.sum(self.iou > 0)), 1)
        iou = self.iou[:nb]

        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False
        if s == 'mean':
            res = np.mean(iou, axis=0)
        elif s == 'median':
            res = np.median(iou, axis=0)
        elif is_number(s):
            res = np.sum(iou > float(s), axis=0) / float(nb)
        return res


if __name__ == '__main__':
    avg = AverageMeter()
    avg.update(time=1.1, accuracy=.99)
    avg.update(time=1.0, accuracy=.90)

    print(avg)

    print(avg.time)
    print(avg.time.avg)
    print(avg.time.val)
    print(avg.SS)



