# fpn code

import random
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class FedPnHead(nn.Sequential):

    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4

        lastconv_output_channels = 6 * in_channels

        layers = [
            nn.Linear(in_channels, inter_channels),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(inter_channels, channels),
        ]

        super(FedPnHead, self).__init__(*layers)


class FedPnBinHead(nn.Sequential):

    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4

        lastconv_output_channels = 6 * in_channels

        layers = [
            nn.Linear(in_channels, inter_channels),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(inter_channels, channels),
        ]

        super(FedPnHead, self).__init__(*layers)


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)


def conv1x1(in_channel, out_channel, **kwargs):

    return nn.Conv2d(in_channel, out_channel, kernel_size=1, **kwargs)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
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


class BotteNeck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BotteNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1,  bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.relu = nn.BatchNorm2d(planes*4)
        self.downsample = downsample
        self.stirde = stride

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

        out += residual
        out = self.relu(out)

        return out


class LocalEncoder(nn.Module):

    def __init__(self, block, layer, seq_len):
        self.inplanes = 64
        super(LocalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3**seq_len, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layer[0])
        self.layer2 = self._make_layer(block, 128, layer[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layer[3], stride=2)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes* block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)


    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        c3 = self.layer1(x)
        c4 = self.layer2(c3)
        c5 = self.layer3(c4)

        return c3, c4, c5



class GlobalEncoder(nn.Module):

    def __init__(self, block):
        super(GlobalEncoder, self).__init__()

        self.conv6 = conv3x3(512*block.expansion, 256, stride=2, padding=1) # p6
        self.conv7 = conv3x3(256, 256, stride=2, padding=1) # p7

        self.lateral_layer1 = conv1x1(512*block.expansion, 256)
        self.lateral_layer2 = conv1x1(256*block.expansion, 256)
        self.lateral_layer3 = conv1x1(128*block.expansion, 256)

        self.corr_layer1 = conv3x3(256, 256, stride=1, padding=1) # p4
        self.corr_layer2 = conv3x3(256, 256, stride=1, padding=1) # p3

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample_add(self, x, y):
        _, _, h, w = y.size()
        x_upsampled = F.interpolate(x, [h,w], mode='bilinear', align_corners=True)

        return x_upsampled + y


    def forwad(self, c):
        c3, c4, c5 = c

        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))

        p5 = self.lateral_layer1(c5)
        lat2 = self.lateral_layer2(c4)
        p4 = self._upsample_add(p5, lat2)
        p4 = self.corr_layer1(p4)
        lat3 = self.lateral_layer3(c3)
        p3 = self._upsample_add(p4, lat3)
        p3 = self.corr_layer2(p3)

        return p3, p4, p5, p6, p7


class FFPN(nn.Module):

    def __init__(self, block, layers, seq_len):
        super(FFPN, self).__init__()
        self.loc_encoder = LocalEncoder(block, layers, seq_len)
        self.glo_encoder = GlobalEncoder(block)

    def forward(self, x):
        c = self.loc_encoder(x)
        ps = self.glo_encoder(c)

        return ps



class SimpleNet(nn.Module):

    def __int__(self, block, layer, seq_len):
        super(SimpleNet, self).__init__()
        self.net
        pass





def ffpn(perms, name, seq_len=1, input_dim=600):

    num = int(name[6:])
    if num <= 50:
        return FFPN(BasicBlock, perms, seq_len)
    else:
        return FFPN(BotteNeck, perms, seq_len)


def base_models(modelname, model_dir, pretrained =False):
    import os

    if modelname[:6] == 'resnet':
        modelperms = {'resnet50':[3,4,6,3]}

        model = ffpn(modelperms[modelname], modelname)

        if pretrained:
            if os.path.isdir(model_dir):
                load_dict = torch.load(os.path.join(model_dir, modelname+'.pth'))

        return model

    print(f'check model path or model name: {model_dir}')
    return None



if __name__ == '__main__':

    print('re')

    print('check')
