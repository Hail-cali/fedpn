import os
import torchvision
from utils.coco_utils import get_coco, get_clinet_coco
import utils.presets as presets
import torch
from utils import image_util


def get_dataset(dir_path, name, image_set, transform, client='client_all'):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode='segmentation', **kwargs)
    paths = {
        "voc": (dir_path, torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21),
        "coco": (dir_path, get_coco, 21)
    }
    p, ds_fn, num_classes = paths[name]

    if image_set == 'val':
        ds = ds_fn(p, image_set=image_set, transforms=transform)
    elif image_set == 'train':
        ds = get_clinet_coco(p, image_set=image_set, transforms=transform, cat_type=client)

    return ds, num_classes


def get_transform(train):
    base_size = 520
    crop_size = 480

    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size)


class LoaderPack:

    def __init__(self, args, client=None, dynamic=False, train_loader=None, val_loader=None, test_loader=None,
                 optim=None, crit=None, global_gpu=False):

        self.args = args
        self.client = client
        self.start_epoch = args.start_epoch
        self.cfg_path = self.args.cfg_path

        self.dynamic = dynamic

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optim
        self.criterion = crit

        if global_gpu:
            self.device = self.args.device
        else:
            self.device = self.set_local_gpu(global_gpu)

        self.log_interval = self.args.log_interval

        # self.losses_avg = None
        # self.accuracies_avg = None
    def set_loader(self, config):
        dataset, num_classes = get_dataset(self.args.data_path, self.args.dataset, "train", get_transform(train=True),
                                           client=self.args.client_type)

        dataset_test, _ = get_dataset(self.args.data_path, self.args.dataset, "val", get_transform(train=False))

        if self.args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        self.train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size,
            sampler=train_sampler, num_workers=self.args.workers,
            collate_fn=image_util.collate_fn, drop_last=True)

        self.val_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=1,
            sampler=test_sampler, num_workers=self.args.workers,
            collate_fn=image_util.collate_fn)

    def set_local_gpu(self, status):
        import torch
        self.args.use_cuda = f'cuda:{self.args.gpu}' if torch.cuda.is_available() else 'cpu'
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
    def local_client_path(self):
        if not self.args.update_status:
            return os.path.join(self.args.resume, f"{self.client}_model_{self.start_epoch}.pth" )
        else:
            return os.path.join(self.args.save_root, self.args.global_model_path)


class GeneratePack(LoaderPack):

    def __init__(self, *args, **kwargs):
        super(GeneratePack, self).__init__(*args, **kwargs)

