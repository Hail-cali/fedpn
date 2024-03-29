
import torch

from model.new_layer import base_models
from torchvision.models.segmentation._utils import _SimpleSegmentationModel

from collections import OrderedDict
from torch import nn
from torch.nn import functional as F


class FedPND(nn.Module):
    __constants__ = ['aux_classifier', 'personal_classifier']

    def __init__(self, backbone, p_net_bone, classifier, aux_classifier=None, personal_classifier=None):
        super(FedPND, self).__init__()
        self.backbone = backbone
        self.p_net_bone = p_net_bone
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier
        self.aux_classifier = aux_classifier
        self.personal_classifier = personal_classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)
        features_personal = self.p_net_bone(x)

        result = OrderedDict()

        x = features["out"]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        result["out"] = x

        x = features_personal["p_net"]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.personal_classifier(x)
        result["p_net"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.aux_classifier(x)
            result["aux"] = x

        return result



class FedPNCif(nn.Module):
    __constants__ = ['aux_classifier', 'global_classifier']

    def __init__(self, backbone, classifier, aux_classifier=None, global_classifier=None):
        super(FedPNCif, self).__init__()
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier
        self.aux_classifier = aux_classifier
        self.global_classifier = global_classifier


    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        # x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.aux_classifier(x)
            # x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        if self.global_classifier is not None:
            x = features["g_net"]
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.global_classifier(x)
            # x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["g_net"] = x

        return result


class FedPN(_SimpleSegmentationModel):

    def __init__(self, global_classifier=None, *args, **kwargs):
        super(FedPN, self).__init__(*args, **kwargs)
        self.global_classifier = global_classifier


    def forward(self,x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        if self.global_classifier is not None:
            x = features["g_net"]
            x = self.global_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["g_net"] = x

        return result





class PN(nn.Module):

    def __init__(self, backbone, ar, head_size, num_classes, bias_head):
        super(PN, self).__init__()

        self.num_classes = num_classes
        self.ar = ar
        self.backbone = backbone
        # self.feature_layers = self.make_features(head_size, bias_head)
        # self.loc_heads = self.make_head(head_size, self.ar*4, bias_head)
        self.conf_heads = self.make_head(head_size, self.ar * num_classes, bias_head)

    def forward(self, x, get_features=False):

        features = self.backbone(x)
        # features = list()
        # for x in sources:
        #     features.append(self.feature_layers(x))

        loc = list()
        conf = list()
        for x in features:
            # loc.append(self.loc_head(x).permute(0, 2, 3, 1).contiguous())
            conf.append(self.conf_heads(x).permute(0, 2, 3, 1).contiguous())

        # loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        return conf.view(conf.size(0), -1, self.num_classes)
        # return loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes)


    def make_features(self, head_size, bias_heads):
        layers = []
        for _ in range(3):
            layers.append(nn.Conv2d(head_size, head_size, kernel_size=3, stride=1))
            layers.append(nn.ReLU(True))

        layers = nn.Sequential(*layers)

        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)

        return layers

    def make_head(self, head_size, out_planes, bias_heads):
        layers =[]

        for _ in range(1):
            layers.append(nn.Conv2d(head_size, head_size, kernel_size=3, stride=1, padding=1,
                                    bias=bias_heads))
            layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(head_size, out_planes, kernel_size=3, stride=1, padding=1))
        layers = nn.Sequential(*layers)
        # for m in layers.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, 0.01)

        return layers

def build_model(modelname, model_dir, ar=9, head_size = 256, num_classes=11, bias_heads=False):

    return PN(base_models(modelname, model_dir), ar, head_size, num_classes, bias_heads)


