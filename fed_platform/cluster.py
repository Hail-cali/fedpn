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
from collections import defaultdict, OrderedDict
from typing import List
import sys
import numpy as np
from tensorboardX import SummaryWriter

TIMEOUT = 3000
OVERFLOW  = 100000003


def fed_aggregate_avg(agg, results, global_layers):
    '''
    :param agg reseted model
    :param results: defaultdict ( model,
    :return:
    '''
    # result['state']
    # result['data_len']
    # result['data_info']

    size = 1
    aux_layer = [str(k) for k in global_layers]
    for client in results:

        for k, v in client['params'].items():
            if k.split('.')[1] in aux_layer and k.split('.')[0] == 'backbone':
                agg[k] += (v * client['data_len'])
                # agg[k] += (v * client['data_len'])
        print(f"debug: client: {client['data_info']['client_name']} | data len: {client['data_len']}")
        size += client['data_len']

    print('debug: ', size, len(results))


    for k, v in agg.items():
        if torch.is_tensor(v):
            agg[k] = torch.div(v, size)

        elif isinstance(v, np.ndarray):
            agg[k] = np.divide(v, size)

        elif isinstance(v, list):
            agg[k] = [val / size for val in agg[k]]

        elif isinstance(v, int):
            agg[k] = v / size


def fed_aggregate_(result):

    return


class Cluster:
    '''
    TEST Cluster
    dynamic allocation model, dataset on Cluster

    :: update model state_dict on Cluster's work module : Trainer.loss.BaseTrainer.update_state_dict_full
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

        self.model = model_map_location(pretrained=self.pack.args.pretrained, global_loss=self.pack.args.global_loss)
        self.model.to(self.pack.device)

        if self.pack.args.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        model_without_ddp = self.model

        # currently not supported
        if self.pack.args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[0,1])
            # self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.pack.args.gpu])
            model_without_ddp = self.model.module

        params_to_optimize = [
            {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
        ]
        if self.pack.args.aux_loss:
            params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
            params_to_optimize.append({"params": params, "lr": config.lr * 10})

        if self.pack.args.global_loss:
            params = [p for p in model_without_ddp.global_classifier.parameters() if p.requires_grad]
            params_to_optimize.append({"params": params, "lr": config.lr * 10})

        self.pack.optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=config.lr, momentum=config.momentum, weight_decay=config.wd)

        self.pack.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.pack.optimizer,
            lambda x: (1 - x / (len(self.pack.data_loader) * self.pack.args.epochs)) ** 0.9)

        self.pack.criterion = seg_criterion

        if self.pack.client != 'global':
            self.update_state_dict_full()

            if self.pack.args.freeze_cls:
                print(f'-- Freeze CLS')
                self.freeze_cls()

    def shared_train(self):
        config = Config(json_path=str(self.pack.cfg_path))

        if self.pack.dynamic:
            self.pack.set_loader(config)

        self._set_model(self.map_model, config)

        return self.module(self.model, self.pack)

    def update_state_dict_full(self):
        base_model = self.model.state_dict()
        # update full model parmas
        if os.path.exists(self.pack.load_local_client_path) and self.pack.start_epoch != 0:
            checkpoint = torch.load(self.pack.load_local_client_path, map_location='cpu')
            # model.load_state_dict(checkpoint['model'], strict=not pack.args.test_only)
            base_model.update(checkpoint['model'])

            if not self.pack.args.test_only:

                self.pack.optimizer.load_state_dict(checkpoint['optimizer'])
                self.pack.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                self.pack.start_epoch = checkpoint['epoch'] + 1

            print(f':: Updated state dict on CheckPoint({self.pack.start_epoch})')
        else:
            checkpoint = torch.load(self.pack.args.cls_path, map_location='cpu')
            base_model.update(checkpoint['model'])
            print(f':: Initialized on , Start Phase or check resume path: {self.pack.args.resume}, {self.pack.args.cls_path}')

        # update global model params
        if self.pack.update_global_path:
            checkpoint = torch.load(self.pack.update_global_path, map_location='cpu')
            base_model.update(checkpoint['model'])
            print(':: Update Global model on client model')
        else:
            print(f':: Initialized on Client model')

        self.model.load_state_dict(base_model, strict=False)

    def freeze_cls(self):
        for name, child in self.model.named_children():

            for param in child.parameters():
                if name in ['classifier']:
                    param.requires_grad = False

                elif name == 'backbone':
                    pass


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
                 base_net=None, base_cluster=None, writer=None, global_gpu=True, verbose=False):

        super(LocalAPI).__init__()
        self.args = args
        self.verbose = verbose
        print(self.args)

        self.mode = args.mode
        if self.mode == 'val':
            self.args.update_status = True
        self.net = base_net
        self.base_agg = None

        self.cluster: SegmentationCluster = base_cluster
        self.stream = base_steam
        self.reader = base_reader

        readme = args.readme
        readme += f"_{args.model_name}_client({args.num_clients}).aux({args.aux_loss}).global.({args.global_loss})"
        readme += f".freeze({args.freeze_cls}).deploy({args.deploy_cls})_{args.dataset}_save_{args.save_root.split('/')[-1]}"
        if writer:
            self.writer = SummaryWriter(f'runs/{readme}')
        else:
            self.writer = None

        self.loop = asyncio.get_event_loop()

        self.tester = SegmentValidator()
        self.fed_state_dict_ls: List[int] = [4,11,12,13]

        if global_gpu:
            self.set_global_gpu()
            self.global_gpu = True
        else:
            self.global_gpu = False
        print(f'Global GPU Set {self.global_gpu} allocated on ')

        if self.args.single_mode == 0:

            # self.client_map = {0: 'client_animal', 1: 'client_vehicle', 2:'client_almost',
            #       3: 'client_half', 4: 'client_all', 5:'client_obj'}
            self.client_map = {0:'client_without_person',1:'client_without_car', 2:'client_without_dog',3:'client_all'}
            print(self.client_map)
        else:
            self.client_map = {0:'client_all'}
            self.args.num_clients = 1
            print(f'Force single mode on({self.args.single_mode}) | num_clients forced to {self.args.num_clients} | client_all used')

        self.clients = {}
        # chk path if False, make dir on path
        image_util.make_dir(self.args.model_dir)
        image_util.make_dir(self.args.save_root)
        image_util.make_dir(os.path.join(self.args.save_root, 'global_cls'))

    def set_global_gpu(self):
        # self.args.use_cuda = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
        self.args.use_cuda = f'cuda:{self.args.gpu}' if torch.cuda.is_available() else 'cpu'
        self.args.device = torch.device(self.args.use_cuda)
        # print(f'| use {self.args.device} | cur gpu num: {torch.cuda.current_device()} | ')

    async def execute_test(self, tasks):
        finished = await asyncio.gather(*tasks)
        return finished

    async def execute_one_round_local(self, tasks):
        result = []
        finished = await asyncio.wait(tasks)

        for job in finished:
            while job:
                task = job.pop()
                result.append(task.result())
                if self.verbose:
                    print(task.result())

        return result

    def execute_one_round_global(self, tasks):
        result = []
        finished, unfinished = self.loop.run_until_complete(
            asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        )
        for task in finished:
            sample = task.result()
            result.append(sample)
            if self.verbose:
                print(task.result())
        return result

    def test(self):
        from utils.pack import get_dataset, get_transform

        pack = LoaderPack(self.args, dynamic=False, val_loader=None, client='server', global_gpu=self.global_gpu)
        pack.tb_writer = self.writer


        dataset_test, _ = get_dataset(pack.data_path, self.args.dataset, "val", get_transform(train=False))
        if self.args.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        else:
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        val_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=1,
            sampler=test_sampler, num_workers=self.args.workers,
            collate_fn=image_util.collate_fn)

        pack.val_loader = val_loader

        # self.base_agg
        server_model = self.net(pretrained=True)
        model_path = os.path.join(self.args.save_root, self.args.global_model_path)

        # update global model parmas to model
        global_model = torch.load(model_path)
        base_model = server_model.state_dict()
        # print(f"global_model params {global_model['model'].keys()}")

        base_model.update(global_model['model'])
        server_model.load_state_dict(base_model, strict=False)
        server_model.to(self.args.device)
        print(f"{'-'*10}Server TEST Global model{'-'*10} ")
        self.tester(server_model, pack)

    def set_phase(self):
        path = os.path.join(self.args.save_root, 'global_cls')
        cls_path = os.path.join(path, self.args.initial_cls)
        self.args.cls_path = cls_path

        if os.path.exists(cls_path):
            pass

        else:
            print(f'Train Global Aux Classifier on Server')
            pack = LoaderPack(self.args, dynamic=True,  client='global', global_gpu=self.global_gpu)
            pack.tb_writer = self.writer
            pack.cfg_path = os.path.join(self.args.root, 'experiments/coco.global.config.json')
            worker = self.cluster(self.net, pack)
            worker.shared_train()
            worker = None
            pack = None

    def train_phase(self):
        self.clients = self._create_client_local()
        tasks = self.dispense_task()

        if sys.version.split(' ')[0] > '3.8.0':
            result = asyncio.run(self.execute_test(tasks))
        else:
            if self.global_gpu:
                result = self.execute_one_round_global(tasks)
            else:
                result = asyncio.run(self.execute_one_round_local(tasks))

        return result

    def deploy_phase(self, data):
        self.args.start_epoch += 1
        self._aggregation(data, self.args.start_epoch)
        self.args.update_status = True
        self.clients.clear()

    def execute(self):

        total_start = time.time()
        print(f"{'+' * 30}")
        start = self.args.start_epoch
        end = self.args.epochs

        self.set_phase()

        for round in range(start, end):
            print(f"{'::'} Federated Learning Round {round} Start ")
            result = self.train_phase()
            self.deploy_phase(result)
            self.test()

        total_end = time.time()
        print(f'total time {total_end-total_start:.5f}times')
        print(f"{'+' * 30}")
        self.loop.close()

        if self.writer:
            self.writer.close()

    def _generate_base_agg(self, load='cpu'):
        self.base_agg = None
        base_agg = self.net(pretrained=self.args.pretrained)

        if load == 'cpu':
            pass

        base_agg.to(self.args.device)
        base_layers = [str(i) for i, b in enumerate(base_agg.backbone) if i in self.fed_state_dict_ls]

        self.base_agg = OrderedDict(
            dict((layer_k, weight) for layer_k, weight in base_agg.state_dict().items()
                 if layer_k.split('.')[1] in base_layers and layer_k.split('.')[0] == 'backbone'))
        print()

    def _create_client_local(self):
        clients = {}
        for l in range(self.args.num_clients):
            client_pack = LoaderPack(self.args, dynamic=True, client=self.client_map[l], global_gpu=self.global_gpu)
            clients[l] = client_pack
        return clients

    def dispense_task(self):

        gen_stream = self.stream(reader=self.reader(model_map_locate=self.net, cluster=self.cluster),
                                 writer=self.writer)

        for l in range(self.args.num_clients):
            gen_stream.scheduler(pack=self.clients[l])
        tasks = gen_stream.executor()
        return tasks

    def _aggregation(self, data, epoch):
        '''

        :param data: asyio module class
        :param epoch:  cur epoch round

        '''

        self._generate_base_agg() # make self.base_agg
        fed_aggregate_avg(self.base_agg, data, self.fed_state_dict_ls)

        model_path = os.path.join(self.args.save_root, self.args.global_model_path)
        checkpoint = {
            'model': self.base_agg,
            'epoch': epoch,
            'args': self.args
        }
        image_util.save_on_master(
            checkpoint,
            model_path)