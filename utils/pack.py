import os


class LoaderPack:

    def __init__(self, args, client=None, train_loader=None, val_loader=None, test_loader=None,
                 optim=None, crit=None, global_gpu=False):

        self.args = args
        self.client = client
        self.start_epoch = args.start_epoch
        self.cfg_path = self.args.cfg_path

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

    def set_local_gpu(self, status):
        import torch
        self.args.use_cuda = f'cuda:{self.args.gpu}' if torch.cuda.is_available() else 'cpu'
        return torch.device(self.args.use_cuda)

    @property
    def data_loader(self):
        if self.args.mode == 'train':
            return self.train_loader
        elif self.args.mode == 'val':
            return self.val_loader

    @property
    def local_client_path(self):
        return os.path.join(self.args.resume, f"{self.client}_model_{self.start_epoch}.pth" )


class GeneratePack(LoaderPack):

    def __init__(self, *args, **kwargs):
        super(GeneratePack, self).__init__(*args, **kwargs)

