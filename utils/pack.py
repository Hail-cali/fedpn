import os
import torchvision
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
import utils.load
from utils.coco_utils import get_coco, get_clinet_coco
import utils.presets as presets
import torch
from utils import image_util
import random
from torch.utils.data import RandomSampler
random.seed(42)

class LimitSampler(RandomSampler):
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self._seed = 42

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))


        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(self._seed)
            # generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator
        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64,
                                     generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples




def get_dataset(dir_path, name, image_set, transform, client='client_all'):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode='segmentation', **kwargs)

    def cif(*args, **kwargs):
        # return utils.load.CIFAR10Dataset(*args, **kwargs)
        return utils.load.RANDOMCIFAR10Dataset(*args, **kwargs)
        # return utils.load.MappedCIFAR10Dataset(*args, **kwargs)

    paths = {
        "voc": (dir_path, torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21),
        "coco": (dir_path, get_coco, 21),
        "cifar": (dir_path, cif, 10),
    }
    p, ds_fn, num_classes = paths[name]

    if name in ['voc', 'voc_aug']:
        ds = ds_fn(p, image_set='train_noval', transforms=transform)
        return ds, num_classes

    elif name == 'coco':
        if image_set == 'val':
            ds = ds_fn(p, image_set=image_set, transforms=transform)
            # ds = get_clinet_coco(p, image_set=image_set, transforms=transform, cat_type=client)
        elif image_set in ['train', 'global']:
            ds = get_clinet_coco(p, image_set=image_set, transforms=transform, cat_type=client)

    elif name == 'cifar':
        ds = ds_fn(p, image_set=image_set, transform=transform, cat_type=client)

    else:
        ds = ds_fn(p, image_set=image_set, transforms=transform, cat_type=client)

    return ds, num_classes


def get_transform(train):
    base_size = 520
    crop_size = 480

    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size)


def init_force_distributed_mode(args):

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)


def init_distributed_mode(args):
    # print('debug ')
    # print(os.environ.keys())
    # print(torch.cuda.device_count())

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    # args.dist_backend = 'gloo'
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)

    # torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                                      world_size=args.world_size, rank=args.rank)

    torch.distributed.init_process_group(backend=args.dist_backend, init_method='file:///mnt/nfs/sharedfile',
                                         world_size=args.world_size, rank=args.rank)

    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


class LoaderPack:

    def __init__(self, args, client=None, dynamic=False, train_loader=None, val_loader=None, test_loader=None,
                 optim=None, crit=None, global_gpu=False):

        self.args = args
        self.client = client
        self.start_epoch = args.start_epoch
        self.cfg_path = os.path.join(self.args.root, self.args.cfg_path[2:])
        self.data_path = os.path.join(self.args.root, self.args.data_path)
        self.dynamic = dynamic
        self.update_global_status = False if self.args.start_epoch == 0 else True
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optim
        self.criterion = crit
        self.tb_writer = None

        if global_gpu:
            self.device = self.args.device
        else:
            self.device = self.set_local_gpu(global_gpu)

        self.log_interval = self.args.log_interval

        # self.losses_avg = None
        # self.accuracies_avg = None
    def set_loader(self, config):
        if self.args.distributed:
            init_distributed_mode(self.args)
        # init_force_distributed_mode(self.args)

        image_set = 'train'

        if self.client == 'global':
            image_set = 'global'

        dataset, num_classes = get_dataset(self.data_path, self.args.dataset, image_set, get_transform(train=True),
                                           client=self.client)

        dataset_test, _ = get_dataset(self.data_path, self.args.dataset, "val", get_transform(train=False), client=self.client)

        if self.args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        else:
            if self.client in ['server', 'global']:
                train_sampler = LimitSampler(dataset, num_samples=640)
            else:
                train_sampler = torch.utils.data.RandomSampler(dataset)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)


        collate_fn_method = None
        if self.args.tasks =='seg':
            collate_fn_method = image_util.collate_fn



        self.train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size,
            sampler=train_sampler, num_workers=self.args.workers,
            collate_fn=collate_fn_method, drop_last=True)


        self.val_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=1,
            sampler=test_sampler, num_workers=self.args.workers,
            collate_fn=collate_fn_method)

    def set_local_gpu(self, status):
        import torch
        # local_gpu = {'client_animal':0, 'client_vehicle':1, 'client_almost':2,
        #           'client_obj':3, 'client_all':0}

        local_gpu = {}

        self.args.use_cuda = f'cuda:{local_gpu[self.client]}' if torch.cuda.is_available() else 'cpu'
        return torch.device(self.args.use_cuda)

    def flush(self):
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.criterion = None

    @property
    def data_loader(self):
        if self.args.mode == 'train':
            return self.train_loader
        elif self.args.mode == 'val':
            return self.val_loader

    @property
    def load_fed_params(self):
        return os.path.join(self.args.save_root, self.args.global_model_path)

    @property
    def load_local_params(self):
        return os.path.join(self.args.resume, f"{self.client}_model_{self.start_epoch-1}.pth" )




