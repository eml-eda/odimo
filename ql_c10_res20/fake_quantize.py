import argparse
import copy
import multiprocessing
import pathlib
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import quantlib.editing.graphs as qg
import quantlib.editing.editing as qe
from quantlib.algorithms.qalgorithms.qatalgorithms.pact import PACTSGD

from net import resnet20
import utils

MEAN = (0.4914, 0.4822, 0.4465)
# MEAN = (0., 0., 0.)
STD = (0.2023, 0.1994, 0.2010)
# STD = (1., 1., 1.)
NB = 8
INT_MAX = 2 ** (NB - 1) - 1
INT_MIN = -2 ** (NB - 1)


def main(args):
    args.data = pathlib.Path(args.data)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Available training HW
    n_cpus = multiprocessing.cpu_count()
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f'Available system CPU(s): {n_cpus}.')
    print(f'Available system GPU(s): {n_gpus}.')
    device = torch.device(torch.cuda.current_device()) if (n_gpus > 0) else torch.device('cpu')

    # Gather data
    upper_bound = max([(1 - m) / s for m, s in zip(MEAN, STD)])
    lower_bound = min([(0 - m) / s for m, s in zip(MEAN, STD)])
    scale = 2 * max(upper_bound, abs(lower_bound)) / (2 ** NB - 1)  # Assume 8b prec for act

    # Fake-Quantize input
    FqInput = transforms.Lambda(lambda x:
                                scale * torch.clip((x / scale).floor(),
                                                   INT_MIN,
                                                   INT_MAX))

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        FqInput,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        FqInput,
    ])

    data_dir = args.data.parent.parent.parent / 'data'
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                             download=True,
                                             transform=transform_train)

    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                            download=True,
                                            transform=transform_test)

    # Split dataset into train and validation
    train_len = int(len(train_set) * 0.9)
    val_len = len(train_set) - train_len
    # Fix generator seed for reproducibility
    data_gen = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_set, [train_len, val_len], generator=data_gen)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

    # Define model
    model = resnet20()
    qg.lw.quantlib_traverse(model).show()  # Show textual model

    # Maybe load pretrained model
    if args.pretrained is not None:
        state_dict = torch.load(args.pretrained)['state_dict']
        model.load_state_dict(state_dict)

    # Fake-Quantization
    model.eval()
    fp_model = qg.fx.quantlib_symbolic_trace(root=model)
    qg.lw.quantlib_traverse(fp_model).show()  # Show textual model
    f2fconverter = qe.f2f.F2F8bitPACTConverter()
    fp_model.eval()
    fq_model_uninit = f2fconverter(fp_model)
    qg.lw.quantlib_traverse(fq_model_uninit).show()  # Show textual model

    # Initialize quantization params with a single validation epoch
    fq_model_uninit.eval()
    fq_model_init = observe(copy.deepcopy(fq_model_uninit), val_dataset)
    # fq_model_init = fq_model_uninit

    # Maybe move to GPU
    fq_model_init.train()
    fq_model = utils.maybe_use_gpu(copy.deepcopy(fq_model_init), device, n_gpus)

    # DEBUG
    # import debug
    # debug.compare_models_output(train_loader, fq_model_uninit, fq_model_init)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = PACTSGD(fq_model,
                        pact_decay=0.001,
                        lr=args.lr, momentum=args.momentum)

    validate(val_loader, fq_model, criterion, 0, device)
    # Check harmonisation
    # fq_model.eval()
    # check_harmonisation(fq_model)

    # Train
    best_epoch = 0
    epoch_wout_improve = 0
    best_acc1_val = 0
    best_acc1_test = 0
    for epoch in range(args.epochs):
        # Train for one epoch
        train(train_loader, fq_model, criterion, optimizer, epoch, device)
        # Validation
        acc1_val = validate(val_loader, fq_model, criterion, epoch, device)
        # Test
        acc1_test = validate(test_loader, fq_model, criterion, epoch, device)

        # Remember best acc1 and save checkpoint
        is_best = acc1_val > best_acc1_val
        if is_best:
            best_epoch = epoch
            best_acc1_val = acc1_val
            best_acc1_test = acc1_test
            epoch_wout_improve = 0
            print(f'New best Acc_val: {best_acc1_val}')
            print(f'New best Acc_test: {best_acc1_test}')
        else:
            epoch_wout_improve += 1
            print(f'Epoch without improvement: {epoch_wout_improve}')

        utils.save_checkpoint(args.data,
                              {'epoch': epoch,
                               'arch': 'fq_resnet20',
                               'state_dict': fq_model.state_dict(),
                               'best_acc1': best_acc1_val,
                               'optimizer': optimizer.state_dict()},
                              is_best)

        # Early-Stop
        if epoch_wout_improve >= args.patience:
            print(f'Early stopping at epoch {epoch}')
            break

    print('Best Acc_val@1 {0} @ epoch {1}'.format(best_acc1_val, best_epoch))
    print('Test Acc_val@1 {0} @ epoch {1}'.format(best_acc1_test, best_epoch))


def train(loader, model, criterion, optimizer, epoch, device):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    curr_lr = optimizer.param_groups[0]['lr']
    progress = utils.ProgressMeter(
        len(loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f'Epoch: [{epoch}/{args.epochs}]\t'
               f'LR: {curr_lr}\t')

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # if q_optimizer is not None:
        #     q_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if q_optimizer is not None:
        #     q_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(loader, model, criterion, epoch, device):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(
        len(loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(f' * Acc@1 {top1.avg:.6f} Acc@5 {top5.avg:.6f}')

    return top1.avg


def observe(model, dataset):
    from quantlib.editing.graphs.nn.harmonisedadd import HarmonisedAdd
    from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
    # from quantlib.algorithms.qalgorithms.qatalgorithms.pact import NNMODULE_TO_PACTMODULE

    for m in model.modules():
        # if isinstance(m, tuple(NNMODULE_TO_PACTMODULE.values())):
        if isinstance(m, (_QModule, HarmonisedAdd)):
            m.start_observing()

    # Collect statistics
    indices = list(range(0, len(dataset)))
    random.shuffle(indices)
    with torch.no_grad():
        for idx in indices:
            x, _ = dataset[idx]
            x = x.unsqueeze(0)
            _ = model(x)

    for m in model.modules():
        # if isinstance(m, (_QModule, HarmonisedAdd)):
        # if isinstance(m, tuple(NNMODULE_TO_PACTMODULE.values())):
        if isinstance(m, _QModule):
            m.stop_observing()

    return model


def check_harmonisation(model):
    for n, m in model.named_modules():
        if isinstance(m, qg.nn.HarmonisedAdd):
            m.harmonise()
            print(n)
            print(m._input_qmodules[0].scale)
            print(m._input_qmodules[1].scale)
            print(m._output_qmodule.scale)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
    parser.add_argument('data', metavar='DIR',
                        help='path where data will be saved')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--patience', default=3, type=int, metavar='N',
                        help='number of epochs wout improvements to wait'
                             'before early stopping')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lrq', '--learning-rate-q', default=1e-5, type=float,
                        metavar='LR', dest='lrq',
                        help='initial q learning rate')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--pretrained', default=None, metavar='PATH',
                        help='path to pre-trained model')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    args = parser.parse_args()

    main(args)
