# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.bbox_helper import center2corner
from torch.autograd import Variable
from utils.anchors import Anchors


class SiamRPN(nn.Module):
    def __init__(self, anchors=None):
        super(SiamRPN, self).__init__()
        self.anchors = anchors  # anchor_cfg
        self.anchor = Anchors(anchors)
        self.anchor_num = self.anchor.anchor_num
        self.features = None
        self.rpn_model = None

        self.all_anchors = None

    def set_all_anchors(self, image_center, size):
        # cx,cy,w,h
        if not self.anchor.generate_all_anchors(image_center, size):
            return
        all_anchors = self.anchor.all_anchors[1] # cx, cy, w, h
        self.all_anchors = torch.from_numpy(all_anchors).float().cuda()
        self.all_anchors = [self.all_anchors[i] for i in range(4)]

    def feature_extractor(self, x):
        return self.features(x)

    def rpn(self, template, search):
        pred_cls, pred_loc = self.rpn_model(template, search)
        return pred_cls, pred_loc

    def _add_rpn_loss(self, label_cls, label_loc, lable_loc_weight, rpn_pred_cls,
                      rpn_pred_loc):
        '''
        :param compute_anchor_targets_fn: functions to produce anchors' learning targets.
        :param rpn_pred_cls: [B, num_anchors * 2, h, w], output of rpn for classification.
        :param rpn_pred_loc: [B, num_anchors * 4, h, w], output of rpn for localization.
        :return: loss of classification and localization, respectively.
        '''
        rpn_loss_cls = select_cross_entropy_loss(rpn_pred_cls, label_cls)

        rpn_loss_loc = weight_l1_loss(rpn_pred_loc, label_loc, lable_loc_weight)

        # classification accuracy, top1
        acc = torch.zeros(1)  # TODO
        return rpn_loss_cls, rpn_loss_loc, acc

    def run(self, template, search, softmax=False):
        """
        run network
        """
        template_feature = self.feature_extractor(template)
        search_feature = self.feature_extractor(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(template_feature, search_feature)
        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc, template_feature, search_feature

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

        rpn_pred_cls, rpn_pred_loc, template_feature, search_feature = self.run(template, search, softmax=self.training)

        outputs = dict(predict=[], losses=[], accuracy=[])

        outputs['predict'] = [rpn_pred_loc, rpn_pred_cls, template_feature, search_feature]
        if self.training:
            rpn_loss_cls, rpn_loss_loc, rpn_acc = self._add_rpn_loss(label_cls, label_loc, lable_loc_weight,
                                                                     rpn_pred_cls, rpn_pred_loc)
            outputs['losses'] = [rpn_loss_cls, rpn_loss_loc]
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
    if len(select.size()) == 0: return 0
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
