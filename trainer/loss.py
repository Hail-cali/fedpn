import csv
import time
from collections import defaultdict
import tqdm
import torch
from utils.pack import LoaderPack
from utils import image_util
import os
from utils import image_util
import copy
from utils.load import Config



def seg_criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = torch.nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']
    if len(losses) == 2:
        return losses['out'] + 0.5 * losses['aux']
    if len(losses) == 3:
        return losses['out'] + 0.5 * losses['aux'] + 0.2 * losses['g_net']


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

    def __call__(self, *args):
        print('base learner')

        return

    def train_one_epoch(self, model, pack):
        model.train()
        # print(pack.tb_writer)
        metric_logger = image_util.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', image_util.SmoothedValue(window_size=1, fmt='{value}'))
        header = 'Epoch: [{}]'.format(pack.start_epoch)
        i = 0
        c = 1
        for image, target in metric_logger.log_every(pack.data_loader, pack.args.log_interval, header):

            image, target = image.to(pack.device), target.to(pack.device)
            output = model(image)
            loss = pack.criterion(output, target)

            pack.optimizer.zero_grad()
            loss.backward()
            pack.optimizer.step()

            pack.lr_scheduler.step()

            metric_logger.update(loss=loss.item(), lr=pack.optimizer.param_groups[0]["lr"])

            if pack.tb_writer:
                if i%pack.args.log_interval==0 and c <= pack.args.max_log:
                    pack.tb_writer.add_scalar(f"{pack.client}/train_step_loss", round(loss.item(),4), c+(pack.start_epoch*pack.args.max_log))
                    c += 1
            i += 1
        print(metric_logger)

    def evaluate(self, model, pack):

        model.eval()
        config = Config(json_path=str(pack.cfg_path))
        confmat = image_util.ConfusionMatrix(config.num_classes)
        metric_logger = image_util.MetricLogger(delimiter="  ")
        header = 'Test:'
        with torch.no_grad():
            for image, target in metric_logger.log_every(pack.val_loader, 100, header):
                image, target = image.to(pack.device), target.to(pack.device)
                output = model(image)
                output = output['out']

                confmat.update(target.flatten(), output.argmax(1).flatten())

            confmat.reduce_from_all_processes()

        acc_global, acc, iu = confmat.compute()

        if pack.tb_writer:
            pack.tb_writer.add_scalar(f"{pack.client}/mean_IoU", iu.mean().item() * 100, pack.start_epoch+1)
            pack.tb_writer.add_text(f"{pack.client}/IoU", str(['{:.1f}'.format(i) for i in (iu * 100).tolist()]),
                                    pack.start_epoch+1)
            pack.tb_writer.add_scalar(f"{pack.client}/infer_time", metric_logger.total_time, pack.start_epoch+1)

            for name, params in model.named_parameters():
                pack.tb_writer.add_histogram(f"{pack.client}/{name}", params.clone().cpu().data.numpy(),
                                             pack.start_epoch+1)

        return confmat


class SegmentTrainer(BaseLearner):

    def __init__(self):
        super(SegmentTrainer, self).__init__()

    def __call__(self, model, pack):
        '''
        train one epoch
        :param model: allocated model
        :param pack: Generated pack
        :return:
        '''

        start_time = time.time()
        print(f"{'--'*30} \n Client RUN :: {pack.client} START || IN {pack.device}")
        self.train_one_epoch(model, pack)
        confmat = self.evaluate(model, pack)
        print(f"|| Client Validate :: {pack.client} ")
        print(confmat)

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': pack.optimizer.state_dict(),
            'lr_scheduler': pack.lr_scheduler.state_dict(),
            'epoch': pack.start_epoch,
            'args': pack.args
        }

        if pack.client == 'global':
            from collections import OrderedDict
            gls_checkpoint = {'model': OrderedDict()}
            chk_cls = ['classifier', 'aux_classifier', 'global_classifier']

            for k, v in checkpoint['model'].items():
                if pack.args.deploy_cls:
                    if k.split('.')[0] in chk_cls:
                        gls_checkpoint['model'][k] = v
                    elif k.split('.')[0] == 'backbone' and k.split('.')[1] in ['4','11','12','13']:
                        gls_checkpoint['model'][k] = v
                else:
                    gls_checkpoint['model'][k] = v

            image_util.save_on_master(
                gls_checkpoint,
                pack.args.cls_path)
            pack.flush()
            return

        else:
            image_util.save_on_master(
                checkpoint,
                os.path.join(pack.args.save_root, f"{pack.client}_model_{pack.start_epoch}.pth"))

            image_util.save_on_master(
                checkpoint,
                os.path.join(pack.args.save_root, f'{pack.client}_checkpoint.pth'))

        result = defaultdict()

        result['params'] = copy.deepcopy(model.state_dict())
        result['state'] = copy.deepcopy(int(pack.start_epoch) + 1)
        result['data_len'] = copy.deepcopy(len(pack.data_loader))
        result['data_info'] = {'client_name':copy.deepcopy(pack.client), 'cls_info':[]}
        print(f"Client RUN :: {pack.client} END \n {'--'*30}")

        end_time = time.time() - start_time
        if pack.tb_writer:
            pack.tb_writer.add_scalar(f"{pack.client}/epoch_run_time", round(end_time, 1), pack.start_epoch+1)

        pack.flush()  # flush dataloader

        return result


class SegmentValidator(BaseLearner):

    def __init__(self):
        super(SegmentValidator, self).__init__()

    def __call__(self, model, pack):
        res = self.evaluate(model, pack)
        print(f"|| Client Validate :: {pack.client} ")
        print(res)
        return res


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

