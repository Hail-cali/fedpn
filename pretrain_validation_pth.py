import torch
from opt import parse_opt
from model.hailnet import hail_mobilenet_v3_large, hail_mobilenet_v3_small
from utils.pack import LoaderPack
from fed_platform.cluster import Cluster, SegmentationCluster, ClassificationCluster

if __name__ == '__main__':
    args = parse_opt()

    model = hail_mobilenet_v3_small

    args.tasks = 'cif'
    args.dataset = 'cifar'

    client = 'train_pretrained'

    args.data_path = 'dataset/cifar'
    args.cfg_path = './experiments/cifar.cif.config.json'

    # args.deploy_cls = True
    args.aux_loss = True
    args.global_loss = True

    args.readme = 'HAIL.NonFed.'
    args.mode = 'train'
    args.use_cuda = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(args.use_cuda)
    image_set = 'train'



    ep = [2,4,6,8,10]
    sp = [0,2,4,6,8]

    args.epochs = 8
    args.start_epoch = 6


    for i in range(5):
        args.epochs = ep[i]
        args.start_epoch = sp[i]
        pack = LoaderPack(args, dynamic=True, client=client, global_gpu=True)
        cluster = ClassificationCluster(model, pack)
        cluster.local_training()

    print()



