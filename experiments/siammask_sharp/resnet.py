import torch.nn as nn
import torch
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
from models.features import Features

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(Features):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # padding = (2 - stride) + (dilation // 2 - 1)
        padding = 2 - stride
        assert stride==1 or dilation==1, "stride and dilation must have one equals to zero at least"
        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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

        if self.downsample is not None:
            residual = self.downsample(x)

        if out.size() != residual.size():
            print(out.size(), residual.size())
        out += residual

        out = self.relu(out)

        return out



class Bottleneck_nop(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_nop, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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

        if self.downsample is not None:
            residual = self.downsample(x)

        s = residual.size(3)
        residual = residual[:, :, 1:s-1, 1:s-1]

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, layer4=False, layer3=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, # 3
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 31x31, 15x15

        self.feature_size = 128 * block.expansion

        if layer3:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2) # 15x15, 7x7
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x:x # identity

        if layer4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4) # 7x7, 3x3
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x:x  # identity

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=3, stride=stride, bias=False,
                              padding=padding, dilation=dd),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        # layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dd))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        p0 = self.relu(x)
        x = self.maxpool(p0)

        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)

        return p0, p1, p2, p3


class ResAdjust(nn.Module):
    def __init__(self,
            block=Bottleneck,
            out_channels=256,
            adjust_number=1,
            fuse_layers=[2,3,4]):
        super(ResAdjust, self).__init__()
        self.fuse_layers = set(fuse_layers)

        if 2 in self.fuse_layers:
            self.layer2 = self._make_layer(block, 128, 1, out_channels, adjust_number)
        if 3 in self.fuse_layers:
            self.layer3 = self._make_layer(block, 256, 2, out_channels, adjust_number)
        if 4 in self.fuse_layers:
            self.layer4 = self._make_layer(block, 512, 4, out_channels, adjust_number)

        self.feature_size = out_channels * len(self.fuse_layers)


    def _make_layer(self, block, plances, dilation, out, number=1):

        layers = []

        for _ in range(number):
            layer = block(plances * block.expansion, plances, dilation=dilation)
            layers.append(layer)

        downsample = nn.Sequential(
                nn.Conv2d(plances * block.expansion, out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out)
                )
        layers.append(downsample)

        return nn.Sequential(*layers)

    def forward(self, p2, p3, p4):

        outputs = []

        if 2 in self.fuse_layers:
            outputs.append(self.layer2(p2))
        if 3 in self.fuse_layers:
            outputs.append(self.layer3(p3))
        if 4 in self.fuse_layers:
            outputs.append(self.layer4(p4))
        # return torch.cat(outputs, 1)
        return outputs


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

if __name__ == '__main__':
    net = resnet50()
    print(net)
    net = net.cuda()

    var = torch.FloatTensor(1,3,127,127).cuda()
    var = Variable(var)

    net(var)
    print('*************')
    var = torch.FloatTensor(1,3,255,255).cuda()
    var = Variable(var)

    net(var)

