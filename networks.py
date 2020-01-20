import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def pool2x2(x):
    return nn.MaxPool2d(kernel_size=2, stride=2)(x)


def upsample2(x):
    return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)


class BasicBlock(nn.Module):

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
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResFCN(nn.Module):

    def __init__(self, inplane, trunk=True, layers=None, planes=None, zero_init_residual=False):
        super(ResFCN, self).__init__()

        self.sample = None
        if trunk:
            self.sample = pool2x2
        else:
            self.sample = upsample2

        if layers is None:
            layers = [1]*3
        self.inplanes = inplane
        self.layer0 = self._make_layer(planes[0], layers[0])
        self.layer1 = self._make_layer(planes[1], layers[1])
        self.layer2 = self._make_layer(planes[2], layers[2])
        self.layer3 = conv1x1(planes[2], planes[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        l0 = self.layer0(x)
        l0 = self.sample(l0)
        l1 = self.layer1(l0)
        l1 = self.sample(l1)
        l2 = self.layer2(l1)
        l2 = self.sample(l2)
        l3 = self.layer3(l2)
        l3 = self.sample(l3)

        return l3


class FeatureTunk(nn.Module):

    def __init__(self, pretrained=True):
        super(FeatureTunk, self).__init__()

        self.color_extractor = BasicBlock(3, 3)
        self.depth_extractor = BasicBlock(1, 1)
        self.mask_extractor = BasicBlock(1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.dense121 = torchvision.models.densenet.densenet121(pretrained=pretrained).features
        self.dense121.conv0 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, color, depth, mask):

        return self.dense121(torch.cat((self.color_extractor(color), self.depth_extractor(depth), self.mask_extractor(mask)), dim=1))
