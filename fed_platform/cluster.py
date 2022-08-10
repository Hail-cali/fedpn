# python 3.8
import asyncio
import copy

import time
import os

from trainer.loss import seg_criterion, cif_criterion, pmn_criterion
from utils.pack import LoaderPack
from utils.load import Config
from utils import image_util

import torch
from collections import OrderedDict

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

    size = 0
    aux_layer = [str(k) for k in global_layers]

    for client in results:

        for k, v in client['params'].items():
            if  k.split('.')[0] == 'backbone':
                try:
                    agg[k] += (v * client['data_len'])
                except:
                    agg[k] = (v * client['data_len'])


        print(f"debug: client: {client['info']['client_name']} | data len: {client['data_len']}")
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


def fed_aggregate_weighted_avg(agg, results, global_layers):

    weighted_idx = sorted([(num, client['info']['acc1']) for num, client in enumerate(results)], key=lambda x: -x[1])

    # weighted_sum_raw = [round(ele, 1) for ele in np.arange(0.8, 0, -0.2, dtype=np.float64).tolist()]
    weighted_sum = [0.3, 0.2, 0.2, 0.1, 0.1, 0.05,  0.05, 0, 0, 0]
    print(weighted_idx,'\n', weighted_sum)
    divide = sum(weighted_sum)

    aux_layer = [str(k) for k in global_layers]

    for rank_idx, (client_num, _)  in enumerate(weighted_idx):
        client = results[client_num]
        for k, v in client['params'].items():
            if k.split('.')[1] in aux_layer and k.split('.')[0] == 'backbone':
                try :
                    agg[k] += (v * weighted_sum[rank_idx])
                except:
                    agg[k] = (v * weighted_sum[rank_idx])

        print(f"debug: client: {client['info']['client_name']} | data len: {client['info']['acc1']}")


    for k, v in agg.items():
        if torch.is_tensor(v):
            agg[k] = torch.div(v, divide)

        elif isinstance(v, np.ndarray):
            agg[k] = np.divide(v, divide)

        elif isinstance(v, list):
            agg[k] = [val / divide for val in agg[k]]

        elif isinstance(v, int):
            agg[k] = v / divide
    return


class Cluster:
    '''
    TEST Cluster
    dynamic allocation model, dataset on Cluster

    '''

    def __init__(self, model_map_location=None, pack=None):
        self.pack: LoaderPack = pack
        self.module = None
        self.map_model = model_map_location
        self.model = None

    def _set_model(self, model_map_location, config):
        # Allocate model, update params
        self.model = model_map_location(pretrained=self.pack.args.pretrained, num_classes=config.num_classes,
                                        global_loss=self.pack.args.global_loss)
        self.model.to(self.pack.device)

        if self.pack.args.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        model_without_ddp = self.model

        # currently not supported
        if self.pack.args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[0,1])
            # self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.pack.args.gpu])
            model_without_ddp = self.model.module

        if self.pack.client not in ['global'] and self.pack.args.freeze_cls:
            self.freeze_cls()

        params_to_optimize = [
            {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
        ]

        if self.pack.args.aux_loss:
            params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
            params_to_optimize.append({"params": params, "lr": config.lr * 10})

        if self.pack.args.global_loss:
            # params = [p for p in model_without_ddp.global_classifier.parameters() if p.requires_grad]
            params_bone = [p for p in model_without_ddp.p_net_bone.parameters() if p.requires_grad]
            params = [p for p in model_without_ddp.personal_classifier.parameters() if p.requires_grad]
            params_to_optimize.append({"params": params, "lr": config.lr * 10})
            params_to_optimize.append({"params": params_bone, "lr": config.lr * 10})


        self.pack.optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=config.lr, momentum=config.momentum, weight_decay=config.wd)

        self.pack.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.pack.optimizer,
            lambda x: (1 - x / (len(self.pack.data_loader) * self.pack.args.epochs)) ** 0.9)

        if self.pack.args.tasks == 'seg':
            self.pack.criterion = seg_criterion

        elif self.pack.args.tasks == 'cif':
            if self.pack.args.model_name == 'FedSMpn':
                self.pack.criterion = cif_criterion
            elif self.pack.args.model_name == 'FedAMpn':
                self.pack.criterion = pmn_criterion

        # # ******************
        if self.pack.client not in ['global', 'train_pretrained']:
            self.update_model()

            if self.pack.args.freeze_cls:
                print(f': -- Freeze CLS [4, 5, 6 ] backbone')
                self.freeze_cls()
        # # ******************
        # # ******************
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params_size = sum([np.prod(p.size()) for p in model_parameters])

        print(f'params size : {params_size}')



    def shared_train(self):
        config = Config(json_path=str(self.pack.cfg_path))

        if self.pack.dynamic:
            self.pack.set_loader(config)

        self._set_model(self.map_model, config)

        return self.module(self.model, self.pack)

    def local_training(self):
        print('Local Train Mode')
        config = Config(json_path=str(self.pack.cfg_path))

        if self.pack.dynamic:
            self.pack.set_loader(config)
            print(len(self.pack.train_loader.dataset))
            print(len(self.pack.val_loader.dataset))

        self._set_model(self.map_model, config)

        for round_ in range(self.pack.args.start_epoch, self.pack.args.epochs):
            self.module(self.model, self.pack)
            self.pack.start_epoch += 1

    def update_model(self):
        base_model = self.model.state_dict()



        # start phase?
        if self.pack.start_epoch == 0:
            if self.pack.args.initial_deploy:
                print(f':: Initialized on Deployed  Initial params ')

            print(f':: Initialized on Randomly')

        else:
            checkpoint = torch.load(self.pack.load_local_params, map_location='cpu')
            base_model.update(checkpoint['model'])
            if not self.pack.args.test_only:
                self.pack.optimizer.load_state_dict(checkpoint['optimizer'])
                self.pack.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                self.pack.start_epoch = checkpoint['epoch'] + 1
            print(f':: Updated state dict on CheckPoint({self.pack.start_epoch})')

        # test mode without Federated learning
        if self.pack.args.diff_exp:
            print(f'Without Federated learning Mode')
            self.model.load_state_dict(base_model, strict=False)
            return

        #Update Federated parmas from global.pth
        if self.pack.update_global_status:
            checkpoint = torch.load(self.pack.load_fed_params, map_location='cpu')
            base_model.update(checkpoint['model'])
            print(f':: Update Federated Global model on {self.pack.client} model {self.pack.load_fed_params}')

        self.model.load_state_dict(base_model, strict=False)

    def freeze_cls(self):
        freeze_layer = ['4','5','6']
        for name,child in self.model.named_children():
            if name == 'backbone':
                for p, param in child.named_parameters():
                    if p.split('.')[0] in freeze_layer:
                        param.requires_grad = False


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
        from trainer.loss import SegmentTrainer, SegmentValidator
        if self.pack.args.mode == 'train':
            self.module = SegmentTrainer()
        else:
            self.module = SegmentValidator()


class ClassificationCluster(Cluster):

    def __init__(self, *args, **kwargs):
        super(ClassificationCluster, self).__init__(*args, **kwargs)
        from trainer.loss import ClassificationTrainer, ClassificationValidator
        if self.pack.args.mode == 'train':
            self.module = ClassificationTrainer()
        else:
            self.module = ClassificationValidator()


class PersonalCluster(Cluster):
    def __init__(self, *args, **kwargs):
        super(PersonalCluster, self).__init__(*args, **kwargs)
        from trainer.loss import PersonalValidator, PersonalTrainer

        if self.pack.args.mode == 'train':
            self.module = PersonalTrainer()
        else:
            self.module = PersonalValidator()


class LocalAPI:

    def __init__(self, args, base_steam=None, base_reader=None,
                 base_net=None, base_cluster=None, writer=None, global_gpu=True, verbose=False):

        super(LocalAPI).__init__()
        self.args = args
        self.verbose = verbose
        print(self.args)

        self.mode = args.mode

        self.net = base_net
        self.base_agg = None

        self.cluster: ClassificationCluster = base_cluster

        self.stream = base_steam
        self.reader = base_reader

        readme = args.readme
        readme += f"-{args.model_name}-client({args.num_clients})-dataset.{args.dataset}({args.save_root.split('/')[-1]})"
        readme += f".aux({args.aux_loss}).global({args.global_loss}).freeze({args.freeze_cls}).deploy({args.initial_deploy})"
        if writer:
            self.writer = SummaryWriter(f'runs/{readme}')
        else:
            self.writer = None

        self.loop = asyncio.get_event_loop()
        self.fed_state_dict_ls: List[int] = [10, 11, 12]

        if global_gpu:
            self.set_global_gpu()
            self.global_gpu = True
        else:
            self.global_gpu = False
        print(f'Global GPU Set {self.global_gpu} allocated on ')

        self.clients = {}
        self.client_map = dict((i, f'c{i + 1}') for i in range(100))
        # if self.args.dataset == 'cifar':
        #     self.client_map = dict((i, f'c{i+1}') for i in range(100))
            # self.client_map = {0: 'c1', 1: 'c2', 2: 'c3', 3: 'c4', 4: 'c5', 5: 'c6', 6: 'c7', 7: 'c8', 8: 'c9', 9: 'c10',
            #                    10: 'c11', 11: 'c12', 12: 'c13', 13: 'c14', 14: 'c15', 15: 'c16', 16: 'c17', 17: 'c18',
            #                    18: 'c19', 19: 'c20', 20: 'c21', 21: 'c22', 22: 'c23', 23: 'c24', 24: 'c25', 25: 'c26',
            #                    26: 'c27', 27: 'c28', 28: 'c29', 29: 'c30', 30: 'c31', 31: 'c32', 32: 'c33', 33: 'c34',
            #                    34: 'c35', 35: 'c36', 36: 'c37', 37: 'c38', 38: 'c39', 39: 'c40', 40: 'c41', 41: 'c42',
            #                    42: 'c43', 43: 'c44', 44: 'c45', 45: 'c46', 46: 'c47', 47: 'c48', 48: 'c49', 49: 'c50',
            #                    50: 'c51', 51: 'c52', 52: 'c53', 53: 'c54', 54: 'c55', 55: 'c56', 56: 'c57', 57: 'c58',
            #                    58: 'c59', 59: 'c60', 60: 'c61', 61: 'c62', 62: 'c63', 63: 'c64', 64: 'c65', 65: 'c66',
            #                    66: 'c67', 67: 'c68', 68: 'c69', 69: 'c70', 70: 'c71', 71: 'c72', 72: 'c73', 73: 'c74',
            #                    74: 'c75', 75: 'c76', 76: 'c77', 77: 'c78', 78: 'c79', 79: 'c80', 80: 'c81', 81: 'c82',
            #                    82: 'c83', 83: 'c84', 84: 'c85', 85: 'c86', 86: 'c87', 87: 'c88', 88: 'c89', 89: 'c90',
            #                    90: 'c91', 91: 'c92', 92: 'c93', 93: 'c94', 94: 'c95', 95: 'c96', 96: 'c97', 97: 'c98',
            #                    98: 'c99', 99: 'c100'}

        image_util.make_dir(self.args.model_dir)
        image_util.make_dir(self.args.save_root)
        image_util.make_dir(os.path.join(self.args.save_root, 'global_cls'))

    def set_global_gpu(self):
        self.args.use_cuda = f'cuda:{self.args.gpu}' if torch.cuda.is_available() else 'cpu'
        self.args.device = torch.device(self.args.use_cuda)

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

    def test_phase(self):

        args = copy.deepcopy(self.args)
        args.start_epoch -= 1
        pack = LoaderPack(args, dynamic=True, client='server', global_gpu=self.global_gpu)
        pack.update_global_status = True
        pack.cfg_path = os.path.join(self.args.root, f"experiments/{pack.args.dataset}.global.config.json")
        pack.tb_writer = self.writer

        cluster = self.cluster(self.net, pack)
        print(f"{'-' * 10}Server TEST Global model{'-' * 10} ")
        result = cluster.shared_train()
        global_model = OrderedDict()
        for k, v in result['params'].items():
            if k.split('.')[0] == 'backbone':
                global_model[k] = v

        model_path = os.path.join(self.args.save_root, self.args.global_model_path)
        checkpoint = {
            'model': global_model,
        }

        image_util.save_on_master(
            checkpoint,
            model_path)

        cluster = None
        pack = None

    def set_phase(self):

        # deploy initial_cls to all client, server
        if self.args.initial_deploy:
            path = os.path.join(self.args.save_root, 'global_cls')
            initial_cls_path = os.path.join(path, self.args.initial_cls)
            self.args.initial_cls_path = initial_cls_path

            if os.path.exists(initial_cls_path):
                pass

            else:
                print(f'Train Global Aux Classifier on Server')
                pack = LoaderPack(self.args, dynamic=True,  client='global', global_gpu=self.global_gpu)
                pack.tb_writer = self.writer
                pack.cfg_path = os.path.join(self.args.root, f"experiments/{pack.args.dataset}.global.config.json")
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
        self.clients.clear()

    def execute(self):

        print(f"{'+' * 30}")
        start = self.args.start_epoch
        end = self.args.epochs

        self.set_phase()

        for round in range(start, end):
            print(f"{'::'} Federated Learning Round {round} Start ")
            result = self.train_phase()
            self.deploy_phase(result)
            self.test_phase()

        print(f"{'+' * 30}")
        self.loop.close()

        if self.writer:
            self.writer.close()

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

        self.base_agg = OrderedDict()
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