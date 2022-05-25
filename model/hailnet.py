from typing import List, Optional

from torchvision.models import mobilenetv3
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation._utils import _SimpleSegmentationModel, _load_weights
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3

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
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    print(out_inplanes)

    global_stage = ['11','12','13']
    global_net = backbone

    aux_classifier = FCNHead(aux_inplanes, num_classes) if aux else None
    classifier = DeepLabHead(out_inplanes, num_classes)
    return DeepLabV3(backbone, classifier, aux_classifier)


def hail_mobilenet_v3_large(
        pretrained: bool = False,
        progress:bool = True,
        num_classes: int = 21,
        pretrained_backbone: bool = True,
        aux_loss: Optional[int] = True):

    if pretrained:
        aux_loss = True
        pretrained_backbone = False

    backbone = mobilenetv3.mobilenet_v3_large(pretrained=pretrained_backbone, dilated=True)
    model = _hail_mobilenetv3(backbone, num_classes, aux_loss)

    if pretrained:
        arch = "deeplabv3_mobilenet_v3_large_coco"
        _load_weights(arch, model, model_urls.get(arch, None), progress)
    return model