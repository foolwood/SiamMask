# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.anchors import Anchors

from models.features import MultiStageFeature
from models.rpn import RPN, DepthCorr
from models.mask import Mask
from utils.load_helper import load_pretrain
from models.resnet import resnet50

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
            params = list(filter(lambda x: x.requires_grad, module.parameters()))
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
        p3 = self.downsample(output[-1])
        return p3

    def forward_all(self, x):
        output = self.features(x)
        p3 = self.downsample(output[-1])
        return output, p3


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
        self.mask = DepthCorr(256, 256, self.oSz ** 2)

    def forward(self, z, x):
        return self.mask(z, x)


class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.v0 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(),
                                nn.Conv2d(16, 4, 3, padding=1), nn.ReLU())

        self.v1 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(),
                                nn.Conv2d(64, 16, 3, padding=1), nn.ReLU())

        self.v2 = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(),
                                nn.Conv2d(128, 32, 3, padding=1), nn.ReLU())

        self.h2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())

        self.h1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())

        self.h0 = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(),
                                nn.Conv2d(4, 4, 3, padding=1), nn.ReLU())

        self.deconv = nn.ConvTranspose2d(256, 32, 15, 15)

        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)

        for modules in [self.v0, self.v1, self.v2, self.h2, self.h1, self.h0, self.deconv, self.post0, self.post1, self.post2, ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.kaiming_uniform_(l.weight, a=1)

    # def forward(self, f, corr_feature, pos=None, test=False):
    #     if test:
    #         p0 = torch.nn.functional.pad(f[0], [16, 16, 16, 16])[:, :, 4 * pos[0]:4 * pos[0] + 61, 4 * pos[1]:4 * pos[1] + 61]
    #         p1 = torch.nn.functional.pad(f[1], [8, 8, 8, 8])[:, :, 2 * pos[0]:2 * pos[0] + 31, 2 * pos[1]:2 * pos[1] + 31]
    #         p2 = torch.nn.functional.pad(f[2], [4, 4, 4, 4])[:, :, pos[0]:pos[0] + 15, pos[1]:pos[1] + 15]
    #     else:
    #         p0 = F.unfold(f[0], (61, 61), padding=0, stride=4).permute(0, 2, 1).contiguous().view(-1, 64, 61, 61)
    #         if not (pos is None): p0 = torch.index_select(p0, 0, pos)
    #         p1 = F.unfold(f[1], (31, 31), padding=0, stride=2).permute(0, 2, 1).contiguous().view(-1, 256, 31, 31)
    #         if not (pos is None): p1 = torch.index_select(p1, 0, pos)
    #         p2 = F.unfold(f[2], (15, 15), padding=0, stride=1).permute(0, 2, 1).contiguous().view(-1, 512, 15, 15)
    #         if not (pos is None): p2 = torch.index_select(p2, 0, pos)
    #
    #     if not (pos is None):
    #         p3 = corr_feature[:, :, pos[0], pos[1]].view(-1, 256, 1, 1)
    #     else:
    #         p3 = corr_feature.permute(0, 2, 3, 1).contiguous().view(-1, 256, 1, 1)
    #
    #     out = self.deconv(p3)
    #     out = self.post0(F.upsample(self.h2(out) + self.v2(p2), size=(31, 31)))
    #     out = self.post1(F.upsample(self.h1(out) + self.v1(p1), size=(61, 61)))
    #     out = self.post2(F.upsample(self.h0(out) + self.v0(p0), size=(127, 127)))
    #     out = out.view(-1, 127 * 127)
    #     return out

    def forward(self, f0, f1, f2, corr_feature):
        p0 = F.unfold(f0, (61, 61), padding=0, stride=4).permute(0, 2, 1).contiguous().view(-1, 64, 61, 61)
        p1 = F.unfold(f1, (31, 31), padding=0, stride=2).permute(0, 2, 1).contiguous().view(-1, 256, 31, 31)
        p2 = F.unfold(f2, (15, 15), padding=0, stride=1).permute(0, 2, 1).contiguous().view(-1, 512, 15, 15)
        p3 = corr_feature.permute(0, 2, 3, 1).contiguous().view(-1, 256, 1, 1)

        out = self.deconv(p3)
        out = self.post0(F.upsample(self.h2(out) + self.v2(p2), size=(31, 31)))
        out = self.post1(F.upsample(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.upsample(self.h0(out) + self.v0(p0), size=(127, 127)))
        out = out.view(-1, 127 * 127)
        return out

    def forward_test(self, f0, f1, f2, corr_feature, pos):
        p0 = torch.nn.functional.pad(f0, [16, 16, 16, 16])[:, :, 4 * pos[0]:4 * pos[0] + 61, 4 * pos[1]:4 * pos[1] + 61]
        p1 = torch.nn.functional.pad(f1, [8, 8, 8, 8])[:, :, 2 * pos[0]:2 * pos[0] + 31, 2 * pos[1]:2 * pos[1] + 31]
        p2 = torch.nn.functional.pad(f2, [4, 4, 4, 4])[:, :, pos[0]:pos[0] + 15, pos[1]:pos[1] + 15]
        p3 = corr_feature[:, :, pos[0], pos[1]].view(-1, 256, 1, 1)

        out = self.deconv(p3)
        out = self.post0(F.upsample(self.h2(out) + self.v2(p2), size=(31, 31)))
        out = self.post1(F.upsample(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.upsample(self.h0(out) + self.v0(p0), size=(127, 127)))
        out = out.view(-1, 127 * 127)
        return out

    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x: x.requires_grad, self.parameters())
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params

class SiamMask(nn.Module):
    def __init__(self, pretrain=False, anchors=None, o_sz=127, g_sz=127):
        super(SiamMask, self).__init__()
        self.anchors = anchors  # anchor_cfg
        self.anchor_num = len(self.anchors["ratios"]) * len(self.anchors["scales"])
        self.anchor = Anchors(anchors)
        self.features = None
        self.rpn_model = None
        self.mask_model = None
        self.o_sz = o_sz
        self.g_sz = g_sz
        self.upSample = nn.Upsample(size=[g_sz, g_sz], mode='bilinear', align_corners=True)
        # self.upSample = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])
        self.features = ResDown(pretrain=pretrain)
        self.rpn_model = UP(anchor_num=self.anchor_num, feature_in=256, feature_out=256)
        self.mask_model = MaskCorr()
        self.refine_model = Refine()

        self.all_anchors = None

    def set_all_anchors(self, image_center, size):
        # cx,cy,w,h
        if not self.anchor.generate_all_anchors(image_center, size):
            return
        all_anchors = self.anchor.all_anchors[1]  # cx, cy, w, h
        self.all_anchors = torch.from_numpy(all_anchors).float().cuda()
        self.all_anchors = [self.all_anchors[i] for i in range(4)]

    def feature_extractor(self, x):
        return self.features(x)

    def rpn(self, template, search):
        pred_cls, pred_loc = self.rpn_model(template, search)
        return pred_cls, pred_loc

    def mask(self, template, search):
        pred_mask = self.mask_model(template, search)
        return pred_mask

    def _add_rpn_loss(self, label_cls, label_loc, lable_loc_weight, label_mask, label_mask_weight,
                      rpn_pred_cls, rpn_pred_loc, rpn_pred_mask):
        rpn_loss_cls = select_cross_entropy_loss(rpn_pred_cls, label_cls)

        rpn_loss_loc = weight_l1_loss(rpn_pred_loc, label_loc, lable_loc_weight)

        rpn_loss_mask, iou_m, iou_5, iou_7 = select_mask_logistic_loss(rpn_pred_mask, label_mask, label_mask_weight)

        return rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_m, iou_5, iou_7

    # def run(self, template, search, softmax=False):
    #     """
    #     run network
    #     """
    #     template_feature = self.feature_extractor(template)
    #     feature, search_feature = self.features.forward_all(search)
    #     rpn_pred_cls, rpn_pred_loc = self.rpn(template_feature, search_feature)
    #     corr_feature = self.mask_model.mask.forward_corr(template_feature, search_feature)  # (b, 256, w, h)
    #     rpn_pred_mask = self.refine_model(feature, corr_feature)
    #
    #     if softmax:
    #         rpn_pred_cls = self.softmax(rpn_pred_cls)
    #     return rpn_pred_cls, rpn_pred_loc, rpn_pred_mask, template_feature, search_feature

    def forward(self, template, search):
        """
        run network
        """
        template_feature = self.feature_extractor(template)
        feature, search_feature = self.features.forward_all(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(template_feature, search_feature)
        corr_feature = self.mask_model.mask.forward_corr(template_feature, search_feature)  # (b, 256, w, h)
        rpn_pred_mask = self.refine_model(feature[0], feature[1], feature[2], corr_feature)

        return rpn_pred_cls, rpn_pred_loc, rpn_pred_mask, template_feature, search_feature

    def softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    # def forward(self, input):
    #     """
    #     :param input: dict of input with keys of:
    #             'template': [b, 3, h1, w1], input template image.
    #             'search': [b, 3, h2, w2], input search image.
    #             'label_cls':[b, max_num_gts, 5] or None(self.training==False),
    #                                  each gt contains x1,y1,x2,y2,class.
    #     :return: dict of loss, predict, accuracy
    #     """
    #     template = input['template']
    #     search = input['search']
    #     if self.training:
    #         label_cls = input['label_cls']
    #         label_loc = input['label_loc']
    #         lable_loc_weight = input['label_loc_weight']
    #         label_mask = input['label_mask']
    #         label_mask_weight = input['label_mask_weight']
    #
    #     rpn_pred_cls, rpn_pred_loc, rpn_pred_mask, template_feature, search_feature = \
    #         self.run(template, search, softmax=self.training)
    #
    #     outputs = dict()
    #
    #     outputs['predict'] = [rpn_pred_loc, rpn_pred_cls, rpn_pred_mask, template_feature, search_feature]
    #
    #     if self.training:
    #         rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_acc_mean, iou_acc_5, iou_acc_7 = \
    #             self._add_rpn_loss(label_cls, label_loc, lable_loc_weight, label_mask, label_mask_weight,
    #                                rpn_pred_cls, rpn_pred_loc, rpn_pred_mask)
    #         outputs['losses'] = [rpn_loss_cls, rpn_loss_loc, rpn_loss_mask]
    #         outputs['accuracy'] = [iou_acc_mean, iou_acc_5, iou_acc_7]
    #
    #     return outputs

    def refine(self, f, pos=None):
        return self.refine_model(f, pos)

    def template(self, template):
        self.zf = self.features(template)

    def track(self, search):
        search = self.features(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, search)
        return rpn_pred_cls, rpn_pred_loc

    def track_mask(self, search):
        self.feature, self.search = self.features.forward_all(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, self.search)
        self.corr_feature = self.mask_model.mask.forward_corr(self.zf, self.search)
        pred_mask = self.mask_model.mask.head(self.corr_feature)
        return rpn_pred_cls, rpn_pred_loc, pred_mask

    def track_refine(self, pos):
        pred_mask = self.refine_model.forward_test(self.feature[0], self.feature[1], self.feature[2], self.corr_feature, pos=pos)
        return pred_mask

def get_cls_loss(pred, label, select):
    if select.nelement() == 0: return pred.sum()*0.
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)

    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = Variable(label.data.eq(1).nonzero().squeeze()).cuda()
    neg = Variable(label.data.eq(0).nonzero().squeeze()).cuda()

    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    """
    :param pred_loc: [b, 4k, h, w]
    :param label_loc: [b, 4k, h, w]
    :param loss_weight:  [b, k, h, w]
    :return: loc loss value
    """
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


def select_mask_logistic_loss(p_m, mask, weight, o_sz=63, g_sz=127):
    weight = weight.view(-1)
    pos = Variable(weight.data.eq(1).nonzero().squeeze())
    if pos.nelement() == 0: return p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0

    if len(p_m.shape) == 4:
        p_m = p_m.permute(0, 2, 3, 1).contiguous().view(-1, 1, o_sz, o_sz)
        p_m = torch.index_select(p_m, 0, pos)
        p_m = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])(p_m)
        p_m = p_m.view(-1, g_sz * g_sz)
    else:
        p_m = torch.index_select(p_m, 0, pos)

    mask_uf = F.unfold(mask, (g_sz, g_sz), padding=0, stride=8)
    mask_uf = torch.transpose(mask_uf, 1, 2).contiguous().view(-1, g_sz * g_sz)

    mask_uf = torch.index_select(mask_uf, 0, pos)
    loss = F.soft_margin_loss(p_m, mask_uf)
    iou_m, iou_5, iou_7 = iou_measure(p_m, mask_uf)
    return loss, iou_m, iou_5, iou_7


def iou_measure(pred, label):
    pred = pred.ge(0)
    mask_sum = pred.eq(1).add(label.eq(1))
    intxn = torch.sum(mask_sum == 2, dim=1).float()
    union = torch.sum(mask_sum > 0, dim=1).float()
    iou = intxn/union
    return torch.mean(iou), (torch.sum(iou > 0.5).float()/iou.shape[0]), (torch.sum(iou > 0.7).float()/iou.shape[0])
    

if __name__ == "__main__":
    p_m = torch.randn(4, 63*63, 25, 25)
    cls = torch.randn(4, 1, 25, 25) > 0.9
    mask = torch.randn(4, 1, 255, 255) * 2 - 1

    loss = select_mask_logistic_loss(p_m, mask, cls)
    print(loss)
