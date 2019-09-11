import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from models.features import Features

__all__ = ['ResNet', 'resnet50']

class Bottleneck(Features):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # padding = (2 - stride) + (dilation // 2 - 1)
        padding = 2 - stride
        assert stride == 1 or dilation == 1, "stride and dilation must have one equals to zero at least"
        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if out.size() != residual.size():
            print(out.size(), residual.size())
        out += residual

        out = self.relu(out)

        return out


def make_downsample_layer(expansion, inplanes, planes, stride, dilation):
    if stride == 1 and dilation == 1:
        return nn.Sequential(
            nn.Conv2d(inplanes, planes * expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * expansion),
        ), dilation
    else:
        if dilation > 1:
            dd = dilation // 2
            padding = dd
        else:
            dd = 1
            padding = 0
        return nn.Sequential(
            nn.Conv2d(inplanes, planes * expansion,
                      kernel_size=3, stride=stride, bias=False,
                      padding=padding, dilation=dd),
            nn.BatchNorm2d(planes * expansion),
        ), dd


class BottleneckWithDownSample(Features):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckWithDownSample, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # padding = (2 - stride) + (dilation // 2 - 1)
        padding = 2 - stride
        self.downsample, dilation = make_downsample_layer(4, inplanes, planes, stride, dilation)
        assert stride == 1 or dilation == 1, "stride and dilation must have one equals to zero at least"
        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.downsample(x)

        if out.size() != residual.size():
            print(out.size(), residual.size())
        out += residual

        out = self.relu(out)

        return out


def make_layer(inplanes, block1, block2, planes, blocks, stride=1, dilation=1):
    layers = []
    layers.append(block2(inplanes, planes, stride, dilation))
    inplanes = planes * block2.expansion
    for i in range(1, blocks):
        layers.append(block1(inplanes, planes, dilation=dilation))

    return nn.Sequential(*layers), inplanes


class ResNet(nn.Module):

    def __init__(self, block1, block2, layers, layer4=False, layer3=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0,  # 3
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1, self.inplanes = make_layer(self.inplanes, block1, block2, 64, layers[0])
        self.layer2, self.inplanes = make_layer(self.inplanes, block1, block2, 128, layers[1], stride=2)  # 31x31, 15x15

        self.feature_size = 128 * block2.expansion

        if layer3:
            self.layer3, self.inplanes = make_layer(self.inplanes, block1, block2, 256, layers[2], stride=1, dilation=2)  # 15x15, 7x7
            self.feature_size = (256 + 128) * block2.expansion
        else:
            self.layer3 = lambda x: x  # identity

        if layer4:
            self.layer4, self.inplanes = make_layer(self.inplanes, block1, block2, 512, layers[3], stride=1, dilation=4)  # 7x7, 3x3
            self.feature_size = 512 * block2.expansion
        else:
            self.layer4 = lambda x: x  # identity

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        p0 = self.relu(x)
        x = self.maxpool(p0)

        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)

        return p0, p1, p2, p3


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, BottleneckWithDownSample, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    return model
