import torch
from opt import parse_opt
from model.hailnet import hail_mobilenet_v3_large, hail_mobilenet_v3_small,hail_mobilenet_v3_pmn
from utils.pack import LoaderPack
from fed_platform.cluster import Cluster, SegmentationCluster, ClassificationCluster, PersonalCluster
from tensorboardX import SummaryWriter
from utils import image_util
import os
from tensorboardX import summary
import numpy as np

if __name__ == '__main__':

    args = parse_opt()


    model = hail_mobilenet_v3_pmn


    args.tasks = 'cif'
    args.dataset ='cifar'
    # args.diff_exp = True

    args.data_path = 'dataset/cifar'

    args.save_root = '/home/hail09/FedPn/experiments/cifar_test_on'
    args.resume =  '/home/hail09/FedPn/experiments/cifar_test_on'
    image_util.make_dir(args.model_dir)
    image_util.make_dir(args.save_root)
    image_util.make_dir(os.path.join(args.save_root, 'global_cls'))

    # args.freeze_cls = True
    args.aux_loss = True
    args.global_loss = True

    args.readme ='DEBUG.'
    args.model_name ='FedAMpn'
    args.mode = 'train'
    args.use_cuda = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(args.use_cuda)


    args.epochs = 10
    args.start_epoch = 0
    image_set = 'train'


    client = 'c1'
    # args.cfg_path = './experiments/cifar.cif.config.json'
    args.cfg_path = './experiments/cifar.global.config.json'
    readme = args.readme
    readme += f"_{args.model_name}_client({client}).aux({args.aux_loss}).global.({args.global_loss})"

    pack = LoaderPack(args, dynamic=True, client=client, global_gpu=True)

    cluster = PersonalCluster(model, pack)
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    cluster.local_training()





