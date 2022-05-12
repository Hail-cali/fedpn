

class LoaderPack:

    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None,
                 optim=None, crit=None):

        self.args = args

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        if self.args.mode == 'train':
            self.data_loader = self.train_loader
        else:
            self.data_loader = self.val_loader

        self.optimizer = optim
        self.criterion = crit
        self.device = self.args.device
        self.log_interval= args.lov_interval

        self.losses_avg = None
        self.accuracies_avg = None
