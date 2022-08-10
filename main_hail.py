

from fed_platform.stream import FedStream, FedReader
from model.hailnet import hail_mobilenet_v3_large, hail_mobilenet_v3_small, hail_mobilenet_v3_pmn
from fed_platform.cluster import LocalAPI, SegmentationCluster, ClassificationCluster, PersonalCluster

from opt import parse_opt
import torch

def seed():

    torch.manual_seed(42)

if __name__ == '__main__':
    args = parse_opt()
    seed()
    model_mapper = None
    cluster_mapper = None

    if args.model_name == 'FedMpn':
        model_mapper = hail_mobilenet_v3_large
    elif args.model_name == 'FedSMpn':
        model_mapper = hail_mobilenet_v3_small
    elif args.model_name == 'FedAMpn':
        model_mapper = hail_mobilenet_v3_pmn

    if args.tasks == 'seg':
        cluster_mapper = SegmentationCluster
    elif args.tasks == 'cif':
        cluster_mapper = PersonalCluster

    running = LocalAPI(args, base_steam=FedStream, base_reader=FedReader, base_net=model_mapper,
                       base_cluster=cluster_mapper, writer=True, global_gpu=args.global_gpu_set, verbose=False)

    running.execute()