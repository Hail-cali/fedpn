

import asyncio
from fed_platform.stream import FedStream, FedReader
from model.hailnet import hail_mobilenet_v3_large, hail_mobilenet_v3_small
from fed_platform.cluster import LocalAPI, SegmentationCluster, ClassificationCluster

from opt import parse_opt


if __name__ == '__main__':
    args = parse_opt()

    model_mapper = None
    cluster_mapper = None

    if args.model_name == 'FedMpn':
        model_mapper = hail_mobilenet_v3_large
    elif args.model_name == 'FedSMpn':
        model_mapper = hail_mobilenet_v3_small

    if args.tasks == 'seg':
        cluster_mapper = SegmentationCluster
    elif args.tasks == 'cif':
        cluster_mapper = ClassificationCluster

    running = LocalAPI(args, base_steam=FedStream, base_reader=FedReader, base_net=model_mapper,
                       base_cluster=cluster_mapper, writer=False, global_gpu=args.global_gpu_set, verbose=False)

    running.execute()