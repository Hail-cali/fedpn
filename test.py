from torchvision.datasets.coco import CocoDetection
# from pycocotools.coco import COCO

from opt import parse_opt
from utils.load import Detection, BaseTransform, Config, ImageDataset, unpickle
import torch
import copy

from model.fedpn import build_model
from torch.utils import data

import torch.optim as optim
import torch.nn as nn
from trainer.loss import *

def main():

    args = parse_opt()
    config = Config(json_path=str(args.cfg_path))

    if args.dataset == 'coco':

        root = args.root + '/' + args.data_path

        data_type = args.data_path.split('/')[-1]
        annFile = f'{root}/annotations/instances_{data_type}.json'

        args.means = [0.485, 0.456, 0.406]
        args.stds = [0.229, 0.224, 0.225]


        # dataset = CocoDetection(root=root, annFile=annFile,
        #                         transform=BaseTransform(config.input_dim, args.means, args.stds))

        dataset = CocoDetection(root=root, annFile=annFile)
        print(dataset)

    if args.dataset == 'cifar':
        root = args.root + '/'

    return


def main2():
    args = parse_opt()
    config = Config(json_path=str(args.cfg_path))


    device = torch.device(f"cuda:{args.gpu}" if args.use_cuda else "cpu")
    print(f"{'*' * 3} set model.{device}() {'*' * 3}")

    file =3
    testmode = True
    result = unpickle(args.file_path, file)

    dataset = ImageDataset(data=result, test_mode=testmode)

    model = build_model(args.basenet, args.model_dir, ar=config.ar, head_size=config.head_size,
                             num_classes=config.num_classes, bias_heads=config.bias_heads)

    train, val = data.random_split(dataset,
                                   [int(len(dataset) * args.train_size),
                                    len(dataset) - int(len(dataset) * args.train_size)])

    train_loader = data.DataLoader(train, batch_size=config.batch_size, shuffle=True)

    val_loader = data.DataLoader(val, batch_size=config.batch_size, shuffle=True)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = dict(epoch=0, train_los=[], train_acc=[], val_los=[], val_acc=[],
                   params=copy.deepcopy(model.state_dict()))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    best_params = dict(best_params=best_model_wts, best_loss=best_loss)

    print('here')

    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, history['epoch'],
                                             args.log_interval, device)

    history['train_los'].append(train_loss), history['train_acc'].append(train_acc)

    val_loss, val_acc = val_epoch(model, val_loader, criterion, device)

    history['val_los'].append(val_loss), history['val_acc'].append(val_acc)

if __name__ == '__main__':

    main2()


    print()