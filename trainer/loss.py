import csv
import time
from collections import defaultdict
import tqdm
import torch
from utils.pack import LoaderPack

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
        return losses['out'] + 0.4 * losses['aux'] + 0.3 * losses['g_net']


def cif_criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():

        losses[name] = torch.nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']
    if len(losses) == 2:
        return losses['out'] + 0.5 * losses['aux']
    if len(losses) == 3:
        return losses['out'] + 0.3 * losses['aux'] + 0.2 * losses['g_net']


def pmn_criterion(inputs, target, bin=False):
    losses = {}

    for name, x in inputs.items():

        losses[name] = torch.nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 2:
        return losses['out'] + losses['p_net']

    if len(losses) == 3:

        return losses['out'] + 0.3 * losses['p_net'] + 0.3 * losses['aux']


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
        pass

    def train_one_epoch(self, model, pack):
        model.train()
        start_time = time.time()

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
                    pack.tb_writer.add_scalar(f"Train_loss(step)/{pack.client}", round(loss.item(), 4), c+(pack.start_epoch*pack.args.max_log))
                    c += 1
            i += 1

        print(metric_logger)
        end_time = time.time() - start_time
        if pack.tb_writer:
            pack.tb_writer.add_scalar(f"Train_time/{pack.client}", round(end_time, 1), pack.start_epoch)

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
                    elif k.split('.')[0] == 'backbone' and k.split('.')[1] in ['4', '5', '6', '10', '11', '12', '13']:
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


class ClassificationLearner(BaseLearner):

    def __init__(self):
        super(ClassificationLearner, self).__init__()

    def evaluate(self, model, pack):
        model.eval()
        config = Config(json_path=str(pack.cfg_path))
        confmat = image_util.ConfusionMatrixAcc(config.num_classes)

        metric_logger = image_util.MetricLogger(delimiter="  ")
        header = 'Test:'
        with torch.no_grad():
            for image, target in metric_logger.log_every(pack.val_loader, 1000, header):
                image, target = image.to(pack.device), target.to(pack.device)
                output = model(image)

                output = output['out']
                acc1, acc5 = confmat.accuracy(output, target, topk=(1, 5))
                batch_size = image.shape[0]
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
                confmat.update(target.flatten(), output.argmax(1).flatten())

            confmat.reduce_from_all_processes()

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()

        print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
                  .format(top1=metric_logger.acc1, top5=metric_logger.acc5))

        acc_global, acc, top_5, acc_class = confmat.compute()

        if pack.tb_writer:
            pack.tb_writer.add_scalar(f"TopAcc@1/{pack.client}", metric_logger.acc1.global_avg, pack.start_epoch)
            pack.tb_writer.add_scalar(f"TopAcc@5/{pack.client}", metric_logger.acc5.global_avg, pack.start_epoch)
            # pack.tb_writer.add_scalar(f"Infer_time/{pack.client}", metric_logger.total_time, pack.start_epoch)
            pack.tb_writer.add_text(f"TopAcc_Class@5/{pack.client}", str(top_5), pack.start_epoch)
            pack.tb_writer.add_text(f"TotalAcc_Class/{pack.client}", str(acc_class), pack.start_epoch)

            # for name, params in model.named_parameters():
            #     if name.split('.')[1] in ['4','11','12']:
            #         pack.tb_writer.add_histogram(f"{name}/{pack.client}", params.clone().cpu().data.numpy(),
            #                                      pack.start_epoch)

        return confmat


class ClassificationTrainer(ClassificationLearner):

    def __init__(self):
        super(ClassificationTrainer, self).__init__()

    def __call__(self, model, pack):
        '''
        train one epoch
        :param model: allocated model
        :param pack: Generated pack
        :return:
        '''
        print(f"{'--'*30} \n Client RUN :: {pack.client} START || IN {pack.device}")

        iter = 1
        if pack.client == 'global':
            iter = 15

        for i in range(iter):
            self.train_one_epoch(model, pack)

        confmat = self.evaluate(model, pack)
        acc_global, acc, top_5, acc_class = confmat.compute()
        print(f"|| Client Validate :: {pack.client} ")
        print(confmat)
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': pack.optimizer.state_dict(),
            'lr_scheduler': pack.lr_scheduler.state_dict(),
            'epoch': pack.start_epoch,
            'args': pack.args
        }

        # Initial Fed STart, deploy cls to clients (only use first SET PHASE)
        if pack.client == 'global':
            from collections import OrderedDict
            gls_checkpoint = {'model': OrderedDict()}
            chk_cls = ['classifier', 'aux_classifier', 'global_classifier']

            for k, v in checkpoint['model'].items():
                gls_checkpoint['model'][k] = v
                # if pack.args.deploy_cls:
                #     if k.split('.')[0] in chk_cls:
                #         gls_checkpoint['model'][k] = v
                #     elif k.split('.')[0] == 'backbone' and k.split('.')[1] in ['4', '5', '6',  '10', '11','12','13']:
                #         gls_checkpoint['model'][k] = v
                # else:
                #     pass

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
        result['info'] = {'client_name':copy.deepcopy(pack.client), 'acc1':acc_global}
        print(f"Client RUN :: {pack.client} END \n {'--'*30}")

        if not pack.args.diff_exp:
            pack.flush()  # flush dataloader

        return result


class ClassificationValidator(ClassificationLearner):

    def __init__(self):
        super(ClassificationValidator, self).__init__()

    def __call__(self, model, pack):
        confmat = self.evaluate(model, pack)
        print(f"|| Client Validate :: {pack.client} ")
        print(confmat)
        return confmat


class PersonalLearner(BaseLearner):

    def __init__(self):
        super(PersonalLearner, self).__init__()

    def train_one_epoch(self, model, pack):
        model.train()
        start_time = time.time()

        metric_logger = image_util.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', image_util.SmoothedValue(window_size=1, fmt='{value}'))
        header = 'Epoch: [{}]'.format(pack.start_epoch)
        i = 0
        c = 1
        for image, target in metric_logger.log_every(pack.data_loader, pack.args.log_interval, header):

            image, target = image.to(pack.device), target.to(pack.device)
            output = model(image)
            if pack.client in ['server', 'global']:
                loss = pack.criterion(output, target)
            else:
                loss = pack.criterion(output, target, bin=True)

            pack.optimizer.zero_grad()
            loss.backward()
            pack.optimizer.step()
            pack.lr_scheduler.step()

            metric_logger.update(loss=loss.item(), lr=pack.optimizer.param_groups[0]["lr"])

            if pack.tb_writer:
                if i % pack.args.log_interval == 0 and c <= pack.args.max_log:
                    pack.tb_writer.add_scalar(f"Train_loss(step)/{pack.client}", round(loss.item(), 4),
                                              c + (pack.start_epoch * pack.args.max_log))
                    c += 1
            i += 1

        print(metric_logger)
        end_time = time.time() - start_time
        # if pack.tb_writer:
        #     pack.tb_writer.add_scalar(f"Train_time/{pack.client}", round(end_time, 1), pack.start_epoch)


    def evaluate(self, model, pack):
        model.eval()
        config = Config(json_path=str(pack.cfg_path))
        confmat = image_util.ConfusionMatrixAcc(config.num_classes)

        metric_logger = image_util.MetricLogger(delimiter="  ")
        header = 'Test:'

        map_loss = 'out'
        if pack.client in ['server', 'global']:
            map_loss = 'out'

        with torch.no_grad():
            for image, target in metric_logger.log_every(pack.val_loader, 1000, header):
                image, target = image.to(pack.device), target.to(pack.device)
                output = model(image)


                output_aux = output['aux']
                output_p_net = output['p_net']
                output = output[map_loss]

                batch_size = image.shape[0]
                if pack.client == 'server':
                    acc1, acc5 = confmat.accuracy(output, target, topk=(1,5))

                    metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                    metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

                else:
                    acc = confmat.accuracy(output, target, topk=(1,))
                    acc1 = acc[0]
                    metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

                p_net_acc = confmat.accuracy(output_p_net, target, topk=(1,))
                p_net_acc = p_net_acc[0]
                aux_acc = confmat.accuracy(output_aux, target, topk=(1,))
                aux_acc = aux_acc[0]
                metric_logger.meters['p_net_acc1'].update(p_net_acc.item(), n=batch_size)
                metric_logger.meters['aux_acc1'].update(aux_acc.item(), n=batch_size)

                confmat.update(target.flatten(), output.argmax(1).flatten())

            confmat.reduce_from_all_processes()

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()

        acc_global, acc, acc_class = confmat.compute()

        if pack.tb_writer:
            pack.tb_writer.add_scalar(f"TopAcc@1/{pack.client}", metric_logger.acc1.global_avg, pack.start_epoch)
            pack.tb_writer.add_scalar(f"TopAcc_p_net@1/{pack.client}", metric_logger.p_net_acc1.global_avg,
                                      pack.start_epoch)
            # pack.tb_writer.add_scalar(f"Infer_time/{pack.client}", metric_logger.total_time, pack.start_epoch)
            # pack.tb_writer.add_text(f"TopAcc_Class@5/{pack.client}", str(top_5), pack.start_epoch)
            if pack.client == 'server':
                pack.tb_writer.add_scalar(f"TopAcc@5/{pack.client}", metric_logger.acc5.global_avg, pack.start_epoch)
                pack.tb_writer.add_text(f"TotalAcc_Class/{pack.client}", str(acc_class), pack.start_epoch)

            # for name, params in model.named_parameters():
            #     if name.split('.')[1] in ['4','11','12']:
            #         pack.tb_writer.add_histogram(f"{name}/{pack.client}", params.clone().cpu().data.numpy(),
            #                                      pack.start_epoch)

        return confmat


class PersonalTrainer(PersonalLearner):

    def __init__(self):
        super(PersonalTrainer, self).__init__()

    def __call__(self, model, pack):
        '''
        train one epoch
        :param model: allocated model
        :param pack: Generated pack
        :return:
        '''
        print(f"{'--'*30} \n Client RUN :: {pack.client} START || IN {pack.device}")

        iter = 1
        if pack.client == 'global':
            iter = 1

        for i in range(iter):
            self.train_one_epoch(model, pack)

        confmat = self.evaluate(model, pack)
        acc_global, acc, acc_class = confmat.compute()
        print(f"|| Client Validate :: {pack.client} ")
        print(confmat)
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': pack.optimizer.state_dict(),
            'lr_scheduler': pack.lr_scheduler.state_dict(),
            'epoch': pack.start_epoch,
            'args': pack.args
        }

        # Initial Fed STart, deploy cls to clients (only use first SET PHASE)
        if pack.client == 'global':
            from collections import OrderedDict
            gls_checkpoint = {'model': OrderedDict()}

            for k, v in checkpoint['model'].items():
                # if k.split('.')[0] in ['backbone', 'p_net_bone']:
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
        result['info'] = {'client_name':copy.deepcopy(pack.client), 'acc1':acc_global}
        print(f"Client RUN :: {pack.client} END \n {'--'*30}")

        if not pack.args.diff_exp:
            pack.flush()  # flush dataloader

        return result


class PersonalValidator(PersonalLearner):

    def __init__(self):
        super(PersonalValidator, self).__init__()

    def __call__(self, model, pack):
        confmat = self.evaluate(model, pack)
        print(f"|| Client Validate :: {pack.client} ")
        print(confmat)
        return confmat