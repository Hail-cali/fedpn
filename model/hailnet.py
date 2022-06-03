from typing import List, Optional

from torchvision.models import mobilenetv3
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
# from torchvision.models.segmentation._utils import _SimpleSegmentationModel, _load_weights
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3

from model.fedpn import FedPN
import torch.nn as nn
from torch.hub import load_state_dict_from_url

# # from ...utils import _log_api_usage_once

__all__ = ["hail_mobilenet_v3_large"]

# Deeplab based code baseline code git :
# followed  code style & baseline process
#  arc changed

model_urls = {
    "deeplabv3_mobilenet_v3_large_coco": "https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth"}


def _hail_mobilenetv3(
        backbone: mobilenetv3.MobileNetV3,
        num_classes: int,
        aux: Optional[bool],
        g_net: Optional[bool],
):
    backbone = backbone.features

    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    out_inplanes = backbone[out_pos].out_channels
    aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    aux_inplanes = backbone[aux_pos].out_channels
    return_layers = {str(out_pos): "out"}
    if aux:
        print('aux')
        return_layers[str(aux_pos)] = "aux"

    global_stage = [11, 12, 13]
    global_stage_indices = [i for i, b in enumerate(backbone) if i in global_stage]
    g_net_pos = global_stage_indices[0]
    g_net_inplanes = backbone[g_net_pos].out_channels

    if g_net:
        return_layers[str(g_net_pos)] = "g_net"

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(aux_inplanes, num_classes) if aux else None
    global_classifier = FCNHead(g_net_inplanes, num_classes) if g_net else None
    classifier = DeepLabHead(out_inplanes, num_classes)
    return FedPN(global_classifier, backbone, classifier, aux_classifier)
    # return FedPN(backbone, classifier, aux_classifier)
    # return DeepLabV3(backbone, classifier, aux_classifier)




def hail_mobilenet_v3_large(
        pretrained: bool = False,
        progress:bool = True,
        num_classes: int = 21,
        pretrained_backbone: bool = True,
        aux_loss: Optional[int] = True,
        global_loss: Optional[int] = True):

    if pretrained:
        aux_loss = True
        pretrained_backbone = False

    backbone = mobilenetv3.mobilenet_v3_large(pretrained=pretrained_backbone, dilated=True)
    model = _hail_mobilenetv3(backbone, num_classes, aux_loss, global_loss)

    if pretrained:
        arch = "deeplabv3_mobilenet_v3_large_coco"
        _load_weights(arch, model, model_urls.get(arch, None), progress)
    return model





# class _SimpleSegmentationModel(nn.Module):
#     __constants__ = ["aux_classifier"]
#
#     def __init__(self, backbone: nn.Module, classifier: nn.Module, aux_classifier: Optional[nn.Module] = None) -> None:
#         super().__init__()
#         _log_api_usage_once(self)
#         self.backbone = backbone
#         self.classifier = classifier
#         self.aux_classifier = aux_classifier
#
#     def forward(self, x: Tensor) -> Dict[str, Tensor]:
#         input_shape = x.shape[-2:]
#         # contract: features is a dict of tensors
#         features = self.backbone(x)
#
#         result = OrderedDict()
#         x = features["out"]
#         x = self.classifier(x)
#         x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
#         result["out"] = x
#
#         if self.aux_classifier is not None:
#             x = features["aux"]
#             x = self.aux_classifier(x)
#             x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
#             result["aux"] = x
#
#         return result


def _load_weights(arch: str, model: nn.Module, model_url: Optional[str], progress: bool) -> None:
    if model_url is None:
        raise ValueError(f"No checkpoint is available for {arch}")
    state_dict = load_state_dict_from_url(model_url, progress=progress)
    model.load_state_dict(state_dict)