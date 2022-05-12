import csv
from collections import defaultdict
import tqdm
import torch
from utils.pack import LoaderPack

class ComputeAvg(object):

    def __init__(self):

        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_acc(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


class BaseLearner:

    def __init__(self):
        super(BaseLearner, self).__init__()

    def unpack(self, dataset):

        return


class Trainer(BaseLearner):

    def __init__(self):
        super(Trainer, self).__init__()

    def __call__(self, model, pack):

        if not isinstance(pack, LoaderPack):
            print('pack class None')
            return

        model.train()
        train_loss = 0.0
        losses = ComputeAvg()
        accuracies = ComputeAvg()
        batch_idx = 0

        # for (data, targets) in tqdm(data_loader, desc='Train ::'):
        for (data, targets) in pack.data_loader:
            data, targets = data.to(pack.device), targets.to(pack.device)
            outputs = model(data)
            loss = pack.criterion(outputs, targets)
            acc = calculate_acc(outputs, targets)

            train_loss += loss.item()
            losses.update(loss.item(), data.size(0))
            accuracies.update(acc, data.size(0))

            pack.optimizer.zero_grad()
            loss.backward()
            pack.optimizer.step()

            if (batch_idx + 1) % pack.log_interval == 0:
                avg_loss = train_loss / pack.log_interval
                print(
                    f' log e[{pack.epoch}]: [{(batch_idx + 1) * len(data)}/{len(pack.data_loader.dataset)} ({((batch_idx + 1) / len(pack.data_loader)) * 100.0:.0f}%)]\tLoss: {avg_loss:.6f}'
                    )
                train_loss = 0.0

            batch_idx += 1

        print(f"Train Done ({len(pack.data_loader.dataset)}"
              f" samples): Average loss: {losses.avg:.4f}\t"
              f"Acc: {(accuracies.avg * 100):.4f}%")
        pack.losses_avg = losses.avg
        pack.accuracies_avg = accuracies.avg

        return model, pack


class Validator(BaseLearner):

    def __init__(self):
        super(Validator, self).__init__()

    def __call__(self, model, pack):
        model.eval()
        print(f'val start', end=': ')
        losses = ComputeAvg()
        accuracies = ComputeAvg()
        with torch.no_grad():
            for (data, targets) in tqdm(pack.val_loader, desc='val epoch:: '):
                data, targets = data.to(pack.device), targets.to(pack.device)
                outputs = model(data)

                loss = pack.criterion(outputs, targets)
                acc = calculate_acc(outputs, targets)

                losses.update(loss.item(), data.size(0))
                accuracies.update(acc, data.size(0))
                # print('epoch')
        # show info

        print(f"Validation Done ({len(pack.val_loader.dataset)}"
              f" samples): Average loss: {losses.avg:.4f}\t"
              f"Acc: {(accuracies.avg * 100):.4f}%")

        pack.losses_avg = losses.avg
        pack.accuracies_avg = accuracies.avg
        return model, pack







# def train_epoch(model, data_loader, criterion, optimizer, epoch, log_interval, device):
#
#     model.train()
#     train_loss = 0.0
#     losses = ComputeAvg()
#     accuracies = ComputeAvg()
#     batch_idx = 0
#
#     # for (data, targets) in tqdm(data_loader, desc='Train ::'):
#     for (data,targets) in data_loader:
#         data, targets = data.to(device), targets.to(device)
#         outputs = model(data)
#         loss = criterion(outputs, targets)
#         acc = calculate_acc(outputs, targets)
#
#         train_loss += loss.item()
#         losses.update(loss.item(), data.size(0))
#         accuracies.update(acc, data.size(0))
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (batch_idx + 1) % log_interval == 0:
#             avg_loss = train_loss / log_interval
#             print(f' log e[{epoch}]: [{(batch_idx + 1) * len(data)}/{len(data_loader.dataset)} ({((batch_idx + 1)/len(data_loader))*100.0:.0f}%)]\tLoss: {avg_loss:.6f}'
#                 )
#             train_loss = 0.0
#
#         batch_idx += 1
#
#
#     print(f"Train Done ({len(data_loader.dataset)}"
#           f" samples): Average loss: {losses.avg:.4f}\t"
#           f"Acc: {(accuracies.avg*100):.4f}%")
#
#     return losses.avg, accuracies.avg
#
# def val_epoch(model, data_loader, criterion, device):
#     model.eval()
#     print(f'val start', end=': ')
#     losses = ComputeAvg()
#     accuracies = ComputeAvg()
#     with torch.no_grad():
#         for (data, targets) in tqdm(data_loader, desc='val epoch:: '):
#             data, targets = data.to(device), targets.to(device)
#             outputs = model(data)
#
#             loss = criterion(outputs, targets)
#             acc = calculate_acc(outputs, targets)
#
#             losses.update(loss.item(), data.size(0))
#             accuracies.update(acc, data.size(0))
#             # print('epoch')
#     # show info
#
#     print(f"Validation Done ({len(data_loader.dataset)}"
#           f" samples): Average loss: {losses.avg:.4f}\t"
#           f"Acc: {(accuracies.avg * 100):.4f}%")
#
#     return losses.avg, accuracies.avg