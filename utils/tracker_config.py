# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from __future__ import division
from utils.anchors import Anchors


class TrackerConfig(object):
    # These are the default hyper-params for SiamMask
    penalty_k = 0.09
    window_influence = 0.39
    lr = 0.38
    seg_thr = 0.3  # for mask
    windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
    # Params from the network architecture, have to be consistent with the training
    exemplar_size = 127  # input z size
    instance_size = 255  # input x size (search region)
    total_stride = 8
    out_size = 63  # for mask
    base_size = 8
    score_size = (instance_size-exemplar_size)//total_stride+1+base_size
    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    round_dight = 0
    anchor = []

    def update(self, newparam=None, anchors=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
        if anchors is not None:
            if isinstance(anchors, dict):
                anchors = Anchors(anchors)
            if isinstance(anchors, Anchors):
                self.total_stride = anchors.stride
                self.ratios = anchors.ratios
                self.scales = anchors.scales
                self.round_dight = anchors.round_dight
        self.renew()

    def renew(self):
        self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1 + self.base_size
        self.anchor_num = len(self.ratios) * len(self.scales)




