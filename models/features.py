# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch.nn as nn
import logging

logger = logging.getLogger('global')


class Features(nn.Module):
    def __init__(self):
        super(Features, self).__init__()
        self.feature_size = -1

    def forward(self, x):
        raise NotImplementedError

    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x:x.requires_grad, self.parameters())
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params

    def load_model(self, f='pretrain.model'):
        with open(f) as f:
            pretrained_dict = torch.load(f)
            model_dict = self.state_dict()
            print(pretrained_dict.keys())
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            print(pretrained_dict.keys())
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


class MultiStageFeature(Features):
    def __init__(self):
        super(MultiStageFeature, self).__init__()

        self.layers = []
        self.train_num = -1
        self.change_point = []
        self.train_nums = []

    def unfix(self, ratio=0.0):
        if self.train_num == -1:
            self.train_num = 0
            self.unlock()
            self.eval()
        for p, t in reversed(list(zip(self.change_point, self.train_nums))):
            if ratio >= p:
                if self.train_num != t:
                    self.train_num = t
                    self.unlock()
                    return True
                break
        return False

    def train_layers(self):
        return self.layers[:self.train_num]

    def unlock(self):
        for p in self.parameters():
            p.requires_grad = False

        logger.info('Current training {} layers:\n\t'.format(self.train_num, self.train_layers()))
        for m in self.train_layers():
            for p in m.parameters():
                p.requires_grad = True

    def train(self, mode):
        self.training = mode
        if mode == False:
            super(MultiStageFeature,self).train(False)
        else:
            for m in self.train_layers():
                m.train(True)

        return self
