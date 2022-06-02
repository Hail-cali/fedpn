from utils.load import Config
from opt import parse_opt
from model.hailnet import hail_mobilenet_v3_large
from typing import Optional, Dict
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from utils.pack import LoaderPack

def _load_state_dict(path):

    return


def _load_weights(arch: str, model: nn.Module, model_url: Optional[str], progress: bool) -> None:
    if model_url is None:
        raise ValueError(f"No checkpoint is available for {arch}")

    state_dict = _load_state_dict(model_url)

    model.load_state_dict(state_dict)

if __name__ == '__main__':

    args = parse_opt()
    c = Config(args.cfg_path)
    model = hail_mobilenet_v3_large(pretrained=args.pretrained)
    model.to('cuda:0')
    # _load_weights('tmp', model, model_url='./')

    # pack = LoaderPack(args, dynamic=True, client='client_animal', global_gpu=True)
    # config = Config(json_path=str(pack.cfg_path))
    # pack.set_loader(config)
    # print(pack.data_loader)
    print()
    '''
    backbone = model.features
    
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    out_inplanes = backbone[out_pos].out_channels
    aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    aux_inplanes = backbone[aux_pos].out_channels
    
    return_layers = {str(out_pos): "out"}
    # if aux:
    #     return_layers[str(aux_pos)] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    print(out_inplanes)
    
    global_stage = ['11', '12', '13']
    global_net = backbone 
    '''

    print()


