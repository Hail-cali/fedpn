import torch.nn as nn
import torch
# from model.layer import base_models
from model.new_layer import base_models
from torchvision.models.segmentation._utils import _SimpleSegmentationModel


class FedPN(_SimpleSegmentationModel):

    def __init__(self, *args, **kwargs):
        super(FedPN, self).__init__(*args, **kwargs)
        self.global_bb = 'test net'








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


