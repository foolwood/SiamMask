# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

    def template(self, template):
        raise NotImplementedError

    def track(self, search):
        raise NotImplementedError

    def param_groups(self, start_lr, feature_mult=1, key=None):
        if key is None:
            params = filter(lambda x:x.requires_grad, self.parameters())
        else:
            params = [v for k, v in self.named_parameters() if (key in k) and v.requires_grad]
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params


class DepthwiseConv2Group(nn.Module):
    def __init__(self):
        super(DepthwiseConv2Group, self).__init__()

    def forward(self, input, kernel):
        batch, channel = kernel.shape[:2]
        input = input.view(1, batch * channel, input.size(2), input.size(3))  # 1 * (b*c) * k * k
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))  # (b*c) * 1 * H * W
        feature = F.conv2d(input, kernel, groups=batch * channel)
        feature = feature.view(batch, channel, feature.size(2), feature.size(3))
        return feature


class DepthCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthCorr, self).__init__()
        # adjust layer for asymmetrical features
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )

        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )

        self.dw_conv2d_group = DepthwiseConv2Group()

    def forward_corr(self, kernel, input):
        kernel = self.conv_kernel(kernel)
        input = self.conv_search(input)
        return self.dw_conv2d_group(input, kernel)

    def forward(self, kernel, search):
        feature = self.forward_corr(kernel, search)
        out = self.head(feature)
        return out
