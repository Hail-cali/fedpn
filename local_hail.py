import torch
from opt import parse_opt
from model.hailnet import hail_mobilenet_v3_large, hail_mobilenet_v3_small, hail_mobilenet_v3_pmn
from utils.pack import LoaderPack
from fed_platform.cluster import ClassificationCluster, PersonalCluster
from tensorboardX import SummaryWriter
import os
from utils import image_util

if __name__ == '__main__':

    args = parse_opt()

    model = hail_mobilenet_v3_pmn

    args.tasks = 'cif'
    args.dataset ='cifar'
    args.data_path = 'dataset/cifar'

    args.save_root = '/home/hail09/FedPn/experiments/STANDALONE.DIFF'
    args.resume = '/home/hail09/FedPn/experiments/STANDALONE.DIFF'

    args.diff_exp = True
    args.aux_loss = True
    args.global_loss = True


    args.readme = 'STANDALONE.DIFF'
    readme = args.readme
    readme += f"_{args.model_name}.aux({args.aux_loss}).global.({args.global_loss})"

    writer = SummaryWriter(f'runs/{readme}')
    sci_clients = {0: 'c1', 1: 'c2', 2: 'c3',
                                   3: 'c4', 4: 'c5',
                                   5: 'c6',  6:'c7', 7:'c8' ,8:'c9' , 9:'c10'}

    image_util.make_dir(args.model_dir)
    image_util.make_dir(args.save_root)
    image_util.make_dir(os.path.join(args.save_root, 'global_cls'))
    for i in range(0, 1):


        args.gpu = 0
        args.mode = 'train'
        args.gpu = 0
        args.use_cuda = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
        args.device = torch.device(args.use_cuda)
        image_set = 'train'
        args.cfg_path = './experiments/cifar.global.config.json'
        args.epochs = 50

        args.start_epoch = 0
        # client = sci_clients[i]
        client = 'server'

        # path = os.path.join(args.save_root, 'global_cls')
        # initial_cls_path = os.path.join(path, args.initial_cls)
        # args.initial_cls_path = initial_cls_path


        pack = LoaderPack(args, dynamic=True, client=client, global_gpu=True)
        pack.tb_writer = writer
        cluster = PersonalCluster(model, pack)
        cluster.local_training()

    print()

    writer.close()
