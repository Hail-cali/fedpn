from typing import List, Optional

from torchvision.models import mobilenetv3
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
import torch
from model.fedpn import FedPN, FedPNCif
from model.layer import FedPnHead
import torch.nn as nn
from torch.hub import load_state_dict_from_url

# # from ...utils import _log_api_usage_once

__all__ = ["hail_mobilenet_v3_large","hail_mobilenet_v3_small"]

# Deeplab based code baseline code git :
# followed  code style & baseline process
#  arc changed

model_urls = {
    "deeplabv3_mobilenet_v3_large_coco": "https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth",
    "hail_FedMpn_v3_small_cifar": "url/to/add"}


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


def _hail_cif_mobilenetv3(
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
        return_layers[str(aux_pos)] = "aux"

    global_stage = [11, 12, 13]
    global_stage_indices = [i for i, b in enumerate(backbone) if i in global_stage]
    g_net_pos = global_stage_indices[0]
    g_net_inplanes = backbone[g_net_pos].out_channels

    if g_net:
        return_layers[str(g_net_pos)] = "g_net"

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FedPnHead(aux_inplanes, num_classes) if aux else None
    global_classifier = FedPnHead(g_net_inplanes, num_classes) if g_net else None
    classifier = FedPnHead(out_inplanes, num_classes)
    return FedPNCif(backbone, classifier, aux_classifier, global_classifier)



def hail_mobilenet_v3_large(
        pretrained: bool = False,
        progress: bool = True,
        num_classes: int = 21,
        pretrained_backbone: bool = True,
        aux_loss: Optional[int] = True,
        global_loss: Optional[int] = False,
        tasks : Optional[str]  = 'seg'):

    if pretrained:
        aux_loss = True
        global_loss = False
        pretrained_backbone = False
    else:
        pretrained_backbone = True

    backbone = mobilenetv3.mobilenet_v3_large(pretrained=pretrained_backbone, dilated=True)

    if tasks == 'seg':
        model = _hail_mobilenetv3(backbone, num_classes, aux_loss, global_loss)
    elif tasks == 'cif':
        model = _hail_cif_mobilenetv3(backbone, num_classes, aux_loss, global_loss)

    if pretrained:
        arch = "deeplabv3_mobilenet_v3_large_coco"
        _load_weights(arch, model, model_urls.get(arch, None), progress)
    return model



def hail_mobilenet_v3_small(
        pretrained: bool = False,
        progress: bool = True,
        num_classes: int = 21,
        pretrained_backbone: bool = True,
        aux_loss: Optional[int] = True,
        global_loss: Optional[int] = False,
        tasks : Optional[str]  = 'seg'):

    if pretrained:
        aux_loss = True
        global_loss = False
        pretrained_backbone = False
    else:
        pretrained_backbone = True

    backbone = mobilenetv3.mobilenet_v3_small(pretrained=pretrained_backbone, dilated=True)


    model = _hail_cif_mobilenetv3(backbone, num_classes, aux_loss, global_loss)

    if pretrained:
        path = '/home/hail09/FedPn/model/pretrained_pth/val_pretrained.pth'
        chk = torch.load(path, map_location='cpu')
        model.load_state_dict(chk['model'])
        print(f'Upload pretrained Server Model for Validate, Global Fed Layer will be changed')
    return model


def _load_weights(arch: str, model: nn.Module, model_url: Optional[str], progress: bool) -> None:
    if model_url is None:
        raise ValueError(f"No checkpoint is available for {arch}")
    state_dict = load_state_dict_from_url(model_url, progress=progress)
    model.load_state_dict(state_dict)