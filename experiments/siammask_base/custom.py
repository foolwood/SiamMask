from models.siammask import SiamMask
from models.features import MultiStageFeature
from models.rpn import RPN, DepthCorr
from models.mask import Mask
import torch
import torch.nn as nn
from utils.load_helper import load_pretrain
from resnet import resnet50


class ResDownS(nn.Module):
    def __init__(self, inplane, outplane):
        super(ResDownS, self).__init__()
        self.downsample = nn.Sequential(
                nn.Conv2d(inplane, outplane, kernel_size=1, bias=False),
                nn.BatchNorm2d(outplane))

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = 4
            r = -4
            x = x[:, :, l:r, l:r]
        return x


class ResDown(MultiStageFeature):
    def __init__(self, pretrain=False):
        super(ResDown, self).__init__()
        self.features = resnet50(layer3=True, layer4=False)
        if pretrain:
            load_pretrain(self.features, 'resnet.model')

        self.downsample = ResDownS(1024, 256)

        self.layers = [self.downsample, self.features.layer2, self.features.layer3]
        self.train_nums = [1, 3]
        self.change_point = [0, 0.5]

        self.unfix(0.0)

    def param_groups(self, start_lr, feature_mult=1):
        lr = start_lr * feature_mult

        def _params(module, mult=1):
            params = list(filter(lambda x:x.requires_grad, module.parameters()))
            if len(params):
                return [{'params': params, 'lr': lr * mult}]
            else:
                return []

        groups = []
        groups += _params(self.downsample)
        groups += _params(self.features, 0.1)
        return groups

    def forward(self, x):
        output = self.features(x)
        p3 = self.downsample(output[1])
        return p3


class UP(RPN):
    def __init__(self, anchor_num=5, feature_in=256, feature_out=256):
        super(UP, self).__init__()

        self.anchor_num = anchor_num
        self.feature_in = feature_in
        self.feature_out = feature_out

        self.cls_output = 2 * self.anchor_num
        self.loc_output = 4 * self.anchor_num

        self.cls = DepthCorr(feature_in, feature_out, self.cls_output)
        self.loc = DepthCorr(feature_in, feature_out, self.loc_output)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc


class MaskCorr(Mask):
    def __init__(self, oSz=63):
        super(MaskCorr, self).__init__()
        self.oSz = oSz
        self.mask = DepthCorr(256, 256, self.oSz**2)

    def forward(self, z, x):
        return self.mask(z, x)


class Custom(SiamMask):
    def __init__(self, pretrain=False, **kwargs):
        super(Custom, self).__init__(**kwargs)
        self.features = ResDown(pretrain=pretrain)
        self.rpn_model = UP(anchor_num=self.anchor_num, feature_in=256, feature_out=256)
        self.mask_model = MaskCorr()

    def template(self, template):
        self.zf = self.features(template)

    def track(self, search):
        search = self.features(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, search)
        return rpn_pred_cls, rpn_pred_loc

    def track_mask(self, search):
        search = self.features(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, search)
        pred_mask = self.mask(self.zf, search)
        return rpn_pred_cls, rpn_pred_loc, pred_mask

