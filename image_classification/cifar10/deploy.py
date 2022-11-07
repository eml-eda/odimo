import argparse
import copy
import os
import pathlib
import random
import time
import warnings

import numpy as np
import torch
# import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms

from deployment.observer import insert_observers, remove_observers
from deployment.quantization import IntegerizationMode, build_qgraph
import models
import models.quant_module as qm
from models.int_module import FakeIntMultiPrecActivConv2d, FakeIntAvgPool2d, FakeIntAdd

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Deployment')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='res21',
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: res20)')
parser.add_argument('-d', '--dataset', default='None', type=str,
                    help='cifar10 or cifar100')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--patience', default=10, type=int, metavar='N',
                    help='number of epochs wout improvements to wait before early stopping')
parser.add_argument('--step-epoch', default=50, type=int, metavar='N',
                    help='number of epochs to decay learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--val-split', default=0.2, type=float,
                    help='Percentage of training data to be used as validation set')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--lrq', '--learning-rate-q', default=1e-5, type=float,
                    metavar='LR', help='initial q learning rate', dest='lrq')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--arch-cfg', '--ac', default='', type=str, metavar='PATH',
                    help='path to architecture configuration')
parser.add_argument('--pretrained-w', default='', type=str, metavar='PATH',
                    help='path to pretrained weights')
# MR
parser.add_argument('-ft', '--fine-tune', dest='fine_tune', action='store_true',
                    help='use pre-trained weights from search phase')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=False, action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

best_acc1 = 0


def main():
    args = parser.parse_args()
    print(args)

    args.data = pathlib.Path(args.data)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    main_worker(args)


def main_worker(args):
    global best_acc1
    global best_acc1_test
    best_acc1_test = 0

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    num_classes = 10

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_dir = args.data.parent.parent.parent / 'data'

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                             download=True, transform=transform_train)

    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                            download=True, transform=transform_test)

    # Split dataset into train and validation
    train_len = int(len(train_set) * 0.9)
    val_len = len(train_set) - train_len
    # Fix generator seed for reproducibility
    data_gen = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_set, [train_len, val_len], generator=data_gen)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # create model
    print("=> creating model '{}'".format(args.arch))
    if len(args.arch_cfg) > 0:
        if os.path.isfile(args.arch_cfg):
            print("=> loading architecture config from '{}'".format(args.arch_cfg))
        else:
            print("=> no architecture found at '{}'".format(args.arch_cfg))
    model_fn = models.__dict__[args.arch]
    model = model_fn(
        args.arch_cfg, num_classes=num_classes, fine_tune=args.fine_tune)
    pretrained_w = torch.load(args.pretrained_w)['state_dict']
    model.load_state_dict(pretrained_w, strict=False)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # Check that pretrained model is working properly
    validate(test_loader, model, args)

    # Insert observers
    obs_model = insert_observers(copy.deepcopy(model),
                                 target_layers=(model.conv_func,
                                                qm.QuantAvgPool2d,
                                                qm.QuantAdd))
    if args.gpu is not None:
        obs_model = obs_model.cuda(args.gpu)
    obs_model.eval()
    obs_model.harden_weights()

    collect_stats(val_loader, obs_model, args)

    '''
    fakeint_model = build_qgraph(copy.deepcopy(obs_model),
                                 output_classes=10,
                                 target_layers=(model.conv_func,
                                                qm.QuantAvgPool2d,
                                                qm.QuantAdd),
                                 mode=IntegerizationMode.FakeInt)
    fakeint_model = remove_observers(copy.deepcopy(fakeint_model),
                                     target_layers=(FakeIntMultiPrecActivConv2d,
                                                    FakeIntAvgPool2d,
                                                    FakeIntAdd))
    criterion = torch.nn.CrossEntropyLoss()

    if args.gpu is not None:
        fakeint_model = fakeint_model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)

    # group model/quantization parameters
    params, q_params = [], []
    for name, param in fakeint_model.named_parameters():
        if ('clip_val' in name) or ('scale_param' in name):
            q_params += [param]
        else:
            params += [param]

    optimizer = torch.optim.SGD(params, args.lr,
                                weight_decay=args.weight_decay)

    train(train_loader, val_loader, test_loader,
          fakeint_model, criterion, optimizer, args.epochs, args)
    '''

    obs_model.store_hardened_weights()
    int_model = build_qgraph(copy.deepcopy(obs_model),
                             output_classes=10,
                             target_layers=(model.conv_func,
                                            qm.QuantAvgPool2d,
                                            qm.QuantAdd),
                             mode=IntegerizationMode.Int)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        int_model = int_model.cuda(args.gpu)

    # compare_models(test_loader, model, int_model, args)

    validate(test_loader, int_model, args)


def collect_stats(loader, model, args):
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            # images = 255 * images
            model(images)


def compare_models(loader, fq_model, int_model, args):
    fq_model.eval()
    int_model.eval()

    def _act(x):
        return torch.floor(torch.min(  # ReLU127
                    torch.max(torch.tensor(0.), x),
                    torch.tensor(127.))
            )
        # return torch.nn.functional.relu(x)

    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # images = 255 * images
            for image in images:
                image = image.unsqueeze(0)
                # label = target[0]

                # fq_out = fq_model(image)

                # int_out = int_model(image)

                # Input Layer
                fq_act1, act_scale = fq_model.conv1.mix_activ(image)
                fq_out1 = fq_model.conv1.mix_weight(fq_act1, act_scale)
                fq_act2, act_scale = fq_model.backbone.bb_1_0.conv1.mix_activ(fq_out1)

                int_act1 = int_model.conv1.mix_activ(image)
                int_out1 = _act(int_model.conv1.mix_weight(int_act1))

                diff_max = torch.abs(int_out1 - (fq_act2 * 127 / act_scale)).max()

                print(f'Input Layer diff max: {diff_max}')

                # First Depthwise
                fq_out2 = fq_model.backbone.bb_1.depth.mix_weight(fq_act2, act_scale)
                fq_act3, act_scale = fq_model.backbone.bb_1.point.mix_activ(fq_out2)

                # int_act2 = int_model.backbone.bb_1.depth.mix_activ(int_out1)
                int_out2 = _act(int_model.backbone.bb_1.depth.mix_weight(int_out1))

                diff_max = torch.abs(int_out2 - (fq_act3 * 127 / act_scale)).max()

                print(f'First Depthwise diff max: {diff_max}')

                # First Pointwise
                fq_out3 = fq_model.backbone.bb_1.point.mix_weight(fq_act3, act_scale)
                fq_act4, act_scale = fq_model.backbone.bb_2.depth.mix_activ(fq_out3)

                # int_act2 = int_model.backbone.bb_1.depth.mix_activ(int_out1)
                int_out3 = _act(int_model.backbone.bb_1.point.mix_weight(int_out2))

                diff_max = torch.abs(int_out3 - (fq_act4 * 127 / act_scale)).max()

                print(f'First Pointwise diff max: {diff_max}')

                # bb_2-bb_5
                fq_out7 = fq_model.backbone.bb_5(
                    fq_model.backbone.bb_4(
                        fq_model.backbone.bb_3(
                            fq_model.backbone.bb_2(fq_out3))))
                fq_act8, act_scale = fq_model.backbone.bb_6.depth.mix_activ(fq_out7)

                int_out6 = _act(int_model.backbone.bb_5.point(int_model.backbone.bb_5.depth(
                    int_model.backbone.bb_4.point(int_model.backbone.bb_4.depth(
                        int_model.backbone.bb_3.point(int_model.backbone.bb_3.depth(
                            int_model.backbone.bb_2.point(int_model.backbone.bb_2.depth(
                                int_out3)))))))))

                diff_max = torch.abs(int_out6 - (fq_act8 * 127 / act_scale)).max()

                print(f'bb_2-bb_5 diff max: {diff_max}')

                # bb_6-bb_12
                fq_out14 = fq_model.backbone.bb_12(
                    fq_model.backbone.bb_11(
                        fq_model.backbone.bb_10(
                            fq_model.backbone.bb_9(
                                fq_model.backbone.bb_8(
                                    fq_model.backbone.bb_7(
                                        fq_model.backbone.bb_6(
                                            fq_out7)))))))
                fq_act15, act_scale = fq_model.backbone.bb_13.depth.mix_activ(fq_out14)

                int_out12 = _act(int_model.backbone.bb_12.point(int_model.backbone.bb_12.depth(
                    int_model.backbone.bb_11.point(int_model.backbone.bb_11.depth(
                        int_model.backbone.bb_10.point(int_model.backbone.bb_10.depth(
                            int_model.backbone.bb_9.point(int_model.backbone.bb_9.depth(
                                int_model.backbone.bb_8.point(int_model.backbone.bb_8.depth(
                                    int_model.backbone.bb_7.point(int_model.backbone.bb_7.depth(
                                        int_model.backbone.bb_6.point(int_model.backbone.bb_6.depth(
                                            int_out6)))))))))))))))

                diff_max = torch.abs(int_out12 - (fq_act15 * 127 / act_scale)).max()

                print(f'bb_6-bb_12 diff max: {diff_max}')

                # bb13 - pool
                fq_out15 = fq_model.backbone.bb_13(fq_out14)

                # int_out13 = _act(int_model.backbone.bb_13.point(
                #     int_model.backbone.bb_13.depth(int_out12)))
                int_out13 = int_model.backbone.bb_13.point(
                    int_model.backbone.bb_13.depth(int_out12))

                fq_pool, act_scale = fq_model.fc.mix_activ(fq_model.backbone.pool(fq_out15))
                int_pool = _act(int_model.backbone.pool(int_out13))

                diff_max = torch.abs(int_pool - (fq_pool * 127 / act_scale)).max()

                print(f'pool diff max: {diff_max}')


def train(train_loader, val_loader, test_loader, model, criterion, optimizer, epochs, args):
    for epoch in range(epochs):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        curr_lr = optimizer.param_groups[0]['lr']
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1],
            prefix="Epoch: [{}/{}]\t"
                   "LR: {}\t".format(epoch, args.epochs, curr_lr))

        # switch to train mode
        model.train()

        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            # for i in range(10000):
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        validate(val_loader, model, args)
        validate(test_loader, model, args)


def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                criterion = criterion.cuda(args.gpu)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(f' * Acc@1 {top1.avg:.6f} Acc@5 {top5.avg:.6f}')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
