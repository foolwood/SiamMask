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


class SiamMask(nn.Module):
    def __init__(self, anchors=None, o_sz=63, g_sz=127):
        super(SiamMask, self).__init__()
        self.anchors = anchors  # anchor_cfg
        self.anchor_num = len(self.anchors["ratios"]) * len(self.anchors["scales"])
        self.anchor = Anchors(anchors)
        self.features = None
        self.rpn_model = None
        self.mask_model = None
        self.o_sz = o_sz
        self.g_sz = g_sz
        self.upSample = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])

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

    def run(self, template, search, softmax=False):
        """
        run network
        """
        template_feature = self.feature_extractor(template)
        search_feature = self.feature_extractor(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(template_feature, search_feature)
        rpn_pred_mask = self.mask(template_feature, search_feature)  # (b, 63*63, w, h)

        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc, rpn_pred_mask, template_feature, search_feature

    def softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, input):
        """
        :param input: dict of input with keys of:
                'template': [b, 3, h1, w1], input template image.
                'search': [b, 3, h2, w2], input search image.
                'label_cls':[b, max_num_gts, 5] or None(self.training==False),
                                     each gt contains x1,y1,x2,y2,class.
        :return: dict of loss, predict, accuracy
        """
        template = input['template']
        search = input['search']
        if self.training:
            label_cls = input['label_cls']
            label_loc = input['label_loc']
            lable_loc_weight = input['label_loc_weight']
            label_mask = input['label_mask']
            label_mask_weight = input['label_mask_weight']

        rpn_pred_cls, rpn_pred_loc, rpn_pred_mask, template_feature, search_feature = \
            self.run(template, search, softmax=self.training)

        outputs = dict()

        outputs['predict'] = [rpn_pred_loc, rpn_pred_cls, rpn_pred_mask, template_feature, search_feature]

        if self.training:
            rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_acc_mean, iou_acc_5, iou_acc_7 = \
                self._add_rpn_loss(label_cls, label_loc, lable_loc_weight, label_mask, label_mask_weight,
                                   rpn_pred_cls, rpn_pred_loc, rpn_pred_mask)
            outputs['losses'] = [rpn_loss_cls, rpn_loss_loc, rpn_loss_mask]
            outputs['accuracy'] = [iou_acc_mean, iou_acc_5, iou_acc_7]

        return outputs

    def template(self, z):
        self.zf = self.feature_extractor(z)
        cls_kernel, loc_kernel = self.rpn_model.template(self.zf)
        return cls_kernel, loc_kernel

    def track(self, x, cls_kernel=None, loc_kernel=None, softmax=False):
        xf = self.feature_extractor(x)
        rpn_pred_cls, rpn_pred_loc = self.rpn_model.track(xf, cls_kernel, loc_kernel)
        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc


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

    p_m = p_m.permute(0, 2, 3, 1).contiguous().view(-1, 1, o_sz, o_sz)
    p_m = torch.index_select(p_m, 0, pos)
    p_m = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])(p_m)
    p_m = p_m.view(-1, g_sz * g_sz)

    mask_uf = F.unfold(mask, (g_sz, g_sz), padding=32, stride=8)
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
