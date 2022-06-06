

import asyncio
from fed_platform.stream import FedStream, FedReader
from model.hailnet import hail_mobilenet_v3_large
from fed_platform.cluster import LocalAPI, SegmentationCluster

from opt import parse_opt


if __name__ == '__main__':
    args = parse_opt()

    running = LocalAPI(args, base_steam=FedStream, base_reader=FedReader, base_net=hail_mobilenet_v3_large,
                       base_cluster=SegmentationCluster, global_gpu=args.global_gpu_set, verbose=False)

    running.execute()