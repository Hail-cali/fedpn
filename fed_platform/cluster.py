# python 3.8
import asyncio
from opt import parse_opt
import time
import os
from trainer.loss import Trainer, Validator, BaseLearner, SegmentTrainer, SegmentValidator, seg_criterion
from utils.pack import LoaderPack
from utils.load import Config
from utils import image_util
import torch
from collections import defaultdict
from torch import nn
import numpy as np



TIMEOUT = 3000


def fed_aggregate_avg(results):
    '''

    :param results: defaultdict ( model,
    :return:
    '''
    global_layers = {str(): "out"}

    agg = defaultdict()
    size = 1

    for client in results:

        for k, v in client['params'].items():
            agg[k] += (v * client['data_len'])
            size += client['data_len']

    if size > 1:
        size -= 1

    for k, v in agg.items():
        if torch.is_tensor(v):
            agg[k] = torch.div(v, size)

        elif isinstance(v, np.ndarray):
            agg[k] = np.divide(v, size)

        elif isinstance(v, list):
            agg[k] = [val / size for val in agg[k]]

        elif isinstance(v, int):
            agg[k] = v / size

    return agg

        # result['state']
        # result['data_len']
        # result['data_info']


def fed_aggregate_(result):

    return


class Cluster:
    '''
    TEST Cluster
    '''

    def __init__(self, model_map_location=None, pack=None):
        self.pack: LoaderPack = pack   # pre set

        if self.pack.args.mode == 'train':
            self.module = Trainer()
        elif self.pack.args.mode == 'val':
            self.module = Validator()
        else:
            print(f'mode {self.pack.args.mode}::')
            self.module = BaseLearner()
        self.map_model = model_map_location
        self.model = None

    def _set_model(self, model_map_location, config):

        self.model = model_map_location(pretrained=self.pack.args.pretrained)
        self.model.to(self.pack.device)

        if self.pack.args.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        model_without_ddp = self.model

        # currently not supported
        if self.pack.args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.gpu])
            model_without_ddp = self.model.module

        params_to_optimize = [
            {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
        ]
        if self.pack.args.aux_loss:
            params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
            params_to_optimize.append({"params": params, "lr": config.lr * 10})

        self.pack.optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=config.lr, momentum=config.momentum, weight_decay=config.wd)

        self.pack.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.pack.optimizer,
            lambda x: (1 - x / (len(self.pack.data_loader) * args.epochs)) ** 0.9)

        self.pack.criterion = seg_criterion

    async def __aenter__(self, *args):
        config = Config(json_path=str(self.pack.cfg_path))

        if self.pack.dynamic:
            self.pack.set_loader(config)

        self._set_model(self.map_model, config)

        return self.module(self.model, self.pack)

    async def __aexit__(self, exc_type, exc_val, exc_tb):

        pass


class SegmentationCluster(Cluster):

    def __init__(self, *args, **kwargs):
        super(SegmentationCluster, self).__init__(*args, **kwargs)
        if self.pack.args.mode == 'train':
            self.module = SegmentTrainer()
        else:
            self.module = SegmentValidator()


class LocalAPI:

    def __init__(self, args, base_steam=None, base_reader=None,
                 base_net=None, base_cluster=None, global_gpu=True, verbose=False):

        super(LocalAPI).__init__()
        self.args = args
        print(self.args)
        self.mode = args.mode
        self.net = base_net
        self.cluster: Cluster = base_cluster
        self.stream: FedStream = base_steam(reader=base_reader(model_map_locate=self.net, cluster=self.cluster),
                                 writer=None)
        if global_gpu:
            self.set_global_gpu()
            self.global_gpu = True
        else:
            self.global_gpu = False

        self.client_map = {0: 'client_animal', 1: 'client_vehicle', 2:'client_almost',
                  3: 'client_obj', 4: 'client_all'}

        self.clients = {}

        self.verbose = verbose

    def set_global_gpu(self):
        # self.args.use_cuda = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
        self.args.use_cuda = f'cuda' if torch.cuda.is_available() else 'cpu'
        self.args.device = torch.device(args.use_cuda)
        # print(f'| use {self.args.device} | cur gpu num: {torch.cuda.current_device()} | ')

    def create_client_local(self):
        clients = {}

        for l in range(self.args.num_clients):
            # dynamic allocate model, dataset, data_loader, lr, optim
            client_pack = LoaderPack(args, dynamic=True, client=self.client_map[l], global_gpu=self.global_gpu)

            # local static allocation
            # client_pack = LoaderPack(args, clinet=self.client_map[l],
            #                         val_loader=None, test_loader=None,
            #                         optim=None, crit=None,
            #                         global_gpu=self.global_gpu)

            clients[l] = client_pack

        return clients

    def dispense_task(self):

        for l in range(self.args.num_clients):
            self.stream.scheduler(pack=self.clients[l])

        tasks = self.stream.executor()

        return tasks

    def execute_one_round(self, tasks):
        result = []
        loop = asyncio.get_event_loop()
        finished, unfinished = loop.run_until_complete(
            asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        )
        for task in finished:
            result.append(task.result())
            if self.verbose:
                print(task.result())
        print("unfinished:", len(unfinished))
        loop.close()
        return result


    def aggregation(self, data, epoch):
        '''

        :param data: asyio module class
        :param epoch:  cur epoch round

        '''

        model = fed_aggregate_avg(data)

        model_path = os.path.join(self.args.save_root, self.args.global_model_path)
        checkpoint = {
            'model': model.state_dict(),
            'epoch': epoch,
            'args': args
        }
        image_util.save_on_master(
            checkpoint,
            model_path)

    def set_phase(self):
        # local allocation
        self.clients = self.create_client_local()

    def train_phase(self):
        tasks = self.dispense_task()
        result = self.execute_one_round(tasks)
        return result

    def deploy_phase(self, data):
        self.args.start_epoch += 1
        self.aggregation(data, self.args.start_epoch)
        self.args.update_status = True

    def execute(self):

        total_start = time.time()

        start = self.args.start_epoch
        end = self.args.epochs

        for round in range(start, end):

            print(f'round {round} start ')
            self.set_phase()
            result = self.train_phase()
            self.deploy_phase(result)

        total_end = time.time()

        print(f'total time {total_end-total_start:.5f}times')
        print(f"{'+' * 20}")

        return result

    @property
    def status(self):
        msg =[f'Global_gpu: {self.global_gpu}']
        return


if __name__ == '__main__':

    from fed_platform.stream import FedStream, FedReader
    from model.hailnet import hail_mobilenet_v3_large
    args = parse_opt()

    running = LocalAPI(args, base_steam=FedStream, base_reader=FedReader, base_net=hail_mobilenet_v3_large,
                       base_cluster=SegmentationCluster, global_gpu=True, verbose=False)

    # running = LocalAPI(args, base_steam=FedStream, base_reader=FedReader, base_net=hail_mobilenet_v3_large,
    #                    base_cluster=SegmentationCluster, global_gpu=False, verbose=False)

    # print(running)

    running.execute()
