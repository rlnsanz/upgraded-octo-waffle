from flor import Flog
if Flog.flagged():
    flog = Flog()
Flog.flagged() and flog.write({'file_path': 'train.py', 'lsn': 31})
import os
import sys
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR


def train(epoch):
    try:
        if Flog.flagged(option='start_function'):
            flog = Flog()
        Flog.flagged() and flog.write({'file_path': 'train.py', 'lsn': 0})
        Flog.flagged() and flog.write({'start_function': 'train', 'lsn': 1})
        Flog.flagged() and flog.write({'lsn': 2, 'params': [{'0.raw.epoch':
            flog.serialize(epoch, 'epoch')}]})
        net.train()
        for batch_index, (images, labels) in enumerate(cifar100_training_loader
            ):
            Flog.flagged() and flog.write({'start_loop': 33, 'lsn': 4})
            if epoch <= args.warm:
                Flog.flagged() and flog.write({'conditional_fork':
                    '(epoch <= args.warm)', 'lsn': 6})
                warmup_scheduler.step()
            else:
                Flog.flagged() and flog.write({'conditional_fork':
                    'not ((epoch <= args.warm))', 'lsn': 7})
            images = Variable(images)
            Flog.flagged() and flog.write({'locals': [{'images': flog.
                serialize(images, 'images')}], 'lineage':
                'images = Variable(images)', 'lsn': 8})
            labels = Variable(labels)
            Flog.flagged() and flog.write({'locals': [{'labels': flog.
                serialize(labels, 'labels')}], 'lineage':
                'labels = Variable(labels)', 'lsn': 9})
            labels = labels.cuda()
            Flog.flagged() and flog.write({'locals': [{'labels': flog.
                serialize(labels, 'labels')}], 'lineage':
                'labels = labels.cuda()', 'lsn': 10})
            images = images.cuda()
            Flog.flagged() and flog.write({'locals': [{'images': flog.
                serialize(images, 'images')}], 'lineage':
                'images = images.cuda()', 'lsn': 11})
            optimizer.zero_grad()
            outputs = net(images)
            Flog.flagged() and flog.write({'locals': [{'outputs': flog.
                serialize(outputs, 'outputs')}], 'lineage':
                'outputs = net(images)', 'lsn': 12})
            loss = loss_function(outputs, labels)
            Flog.flagged() and flog.write({'locals': [{'loss': flog.
                serialize(loss, 'loss')}], 'lineage':
                'loss = loss_function(outputs, labels)', 'lsn': 13})
            loss.backward()
            optimizer.step()
            n_iter = (epoch - 1) * len(cifar100_training_loader
                ) + batch_index + 1
            Flog.flagged() and flog.write({'locals': [{'n_iter': flog.
                serialize(n_iter, 'n_iter')}], 'lineage':
                'n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1'
                , 'lsn': 14})
            last_layer = list(net.children())[-1]
            Flog.flagged() and flog.write({'locals': [{'last_layer': flog.
                serialize(last_layer, 'last_layer')}], 'lineage':
                'last_layer = list(net.children())[-1]', 'lsn': 15})
            for name, para in last_layer.named_parameters():
                Flog.flagged() and flog.write({'start_loop': 52, 'lsn': 16})
                if 'weight' in name:
                    Flog.flagged() and flog.write({'conditional_fork':
                        '("weight" in name)', 'lsn': 18})
                    writer.add_scalar('LastLayerGradients/grad_norm2_weights',
                        para.grad.norm(), n_iter)
                else:
                    Flog.flagged() and flog.write({'conditional_fork':
                        'not (("weight" in name))', 'lsn': 19})
                if 'bias' in name:
                    Flog.flagged() and flog.write({'conditional_fork':
                        '("bias" in name)', 'lsn': 20})
                    writer.add_scalar('LastLayerGradients/grad_norm2_bias',
                        para.grad.norm(), n_iter)
                else:
                    Flog.flagged() and flog.write({'conditional_fork':
                        'not (("bias" in name))', 'lsn': 21})
                Flog.flagged() and flog.write({'end_loop': 52, 'lsn': 17})
            print(
                'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'
                .format(loss.item(), optimizer.param_groups[0]['lr'], epoch
                =epoch, trained_samples=batch_index * args.b + len(images),
                total_samples=len(cifar100_training_loader.dataset)))
            writer.add_scalar('Train/loss', loss.item(), n_iter)
            Flog.flagged() and flog.write({'end_loop': 33, 'lsn': 5})
        for name, param in net.named_parameters():
            Flog.flagged() and flog.write({'start_loop': 69, 'lsn': 22})
            layer, attr = os.path.splitext(name)
            Flog.flagged() and flog.write({'locals': [{'layer': flog.
                serialize(layer, 'layer')}, {'attr': flog.serialize(attr,
                'attr')}], 'lineage':
                'layer, attr = os.path.splitext(name)', 'lsn': 24})
            attr = attr[1:]
            Flog.flagged() and flog.write({'locals': [{'attr': flog.
                serialize(attr, 'attr')}], 'lineage': 'attr = attr[1:]',
                'lsn': 25})
            writer.add_histogram('{}/{}'.format(layer, attr), param, epoch)
            Flog.flagged() and flog.write({'end_loop': 69, 'lsn': 23})
    finally:
        Flog.flagged() and flog.write({'end_function': 'train', 'lsn': 3})
        Flog.flagged(option='end_function') and flog.writer.close()


def eval_training(epoch):
    try:
        if Flog.flagged(option='start_function'):
            flog = Flog()
        Flog.flagged() and flog.write({'file_path': 'train.py', 'lsn': 0})
        Flog.flagged() and flog.write({'start_function': 'eval_training',
            'lsn': 1})
        Flog.flagged() and flog.write({'lsn': 2, 'params': [{'0.raw.epoch':
            flog.serialize(epoch, 'epoch')}]})
        net.eval()
        test_loss = 0.0
        Flog.flagged() and flog.write({'locals': [{'test_loss': flog.
            serialize(test_loss, 'test_loss')}], 'lineage':
            'test_loss = 0.0', 'lsn': 4})
        correct = 0.0
        Flog.flagged() and flog.write({'locals': [{'correct': flog.
            serialize(correct, 'correct')}], 'lineage': 'correct = 0.0',
            'lsn': 5})
        for images, labels in cifar100_test_loader:
            Flog.flagged() and flog.write({'start_loop': 80, 'lsn': 6})
            images = Variable(images)
            Flog.flagged() and flog.write({'locals': [{'images': flog.
                serialize(images, 'images')}], 'lineage':
                'images = Variable(images)', 'lsn': 8})
            labels = Variable(labels)
            Flog.flagged() and flog.write({'locals': [{'labels': flog.
                serialize(labels, 'labels')}], 'lineage':
                'labels = Variable(labels)', 'lsn': 9})
            images = images.cuda()
            Flog.flagged() and flog.write({'locals': [{'images': flog.
                serialize(images, 'images')}], 'lineage':
                'images = images.cuda()', 'lsn': 10})
            labels = labels.cuda()
            Flog.flagged() and flog.write({'locals': [{'labels': flog.
                serialize(labels, 'labels')}], 'lineage':
                'labels = labels.cuda()', 'lsn': 11})
            outputs = net(images)
            Flog.flagged() and flog.write({'locals': [{'outputs': flog.
                serialize(outputs, 'outputs')}], 'lineage':
                'outputs = net(images)', 'lsn': 12})
            loss = loss_function(outputs, labels)
            Flog.flagged() and flog.write({'locals': [{'loss': flog.
                serialize(loss, 'loss')}], 'lineage':
                'loss = loss_function(outputs, labels)', 'lsn': 13})
            test_loss += loss.item()
            Flog.flagged() and flog.write({'locals': [{'test_loss': flog.
                serialize(test_loss, 'test_loss')}], 'lineage':
                'test_loss += loss.item()', 'lsn': 14})
            _, preds = outputs.max(1)
            Flog.flagged() and flog.write({'locals': [{'_': flog.serialize(
                _, '_')}, {'preds': flog.serialize(preds, 'preds')}],
                'lineage': '_, preds = outputs.max(1)', 'lsn': 15})
            correct += preds.eq(labels).sum()
            Flog.flagged() and flog.write({'locals': [{'correct': flog.
                serialize(correct, 'correct')}], 'lineage':
                'correct += preds.eq(labels).sum()', 'lsn': 16})
            Flog.flagged() and flog.write({'end_loop': 80, 'lsn': 7})
        print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
            test_loss / len(cifar100_test_loader.dataset), correct.float() /
            len(cifar100_test_loader.dataset)))
        print()
        writer.add_scalar('Test/Average loss', test_loss / len(
            cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(
            cifar100_test_loader.dataset), epoch)
        __return__ = correct.float() / len(cifar100_test_loader.dataset)
        Flog.flagged() and flog.write({'locals': [{'__return__': flog.
            serialize(__return__, '__return__')}], 'lineage':
            '__return__ = correct.float() / len(cifar100_test_loader.dataset)',
            'lsn': 17})
        return __return__
    finally:
        Flog.flagged() and flog.write({'end_function': 'eval_training',
            'lsn': 3})
        Flog.flagged(option='end_function') and flog.writer.close()


if __name__ == '__main__':
    Flog.flagged() and flog.write({'conditional_fork':
        '(__name__ == "__main__")', 'lsn': 0})
    parser = argparse.ArgumentParser()
    Flog.flagged() and flog.write({'locals': [{'parser': flog.serialize(
        parser, 'parser')}], 'lineage':
        'parser = argparse.ArgumentParser()', 'lsn': 2})
    parser.add_argument('-net', type=str, default='vgg16', help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help=
        'number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help=
        'batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help=
        'whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help=
        'warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help=
        'initial learning rate')
    args = parser.parse_args()
    Flog.flagged() and flog.write({'locals': [{'args': flog.serialize(args,
        'args')}], 'lineage': 'args = parser.parse_args()', 'lsn': 3})
    net = get_network(args, use_gpu=args.gpu)
    Flog.flagged() and flog.write({'locals': [{'net': flog.serialize(net,
        'net')}], 'lineage': 'net = get_network(args, use_gpu=args.gpu)',
        'lsn': 4})
    cifar100_training_loader = get_training_dataloader(settings.
        CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD, num_workers=args.
        w, batch_size=args.b, shuffle=args.s)
    Flog.flagged() and flog.write({'locals': [{'cifar100_training_loader':
        flog.serialize(cifar100_training_loader, 'cifar100_training_loader'
        )}], 'lineage':
        'cifar100_training_loader = get_training_dataloader(settings.    CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD, num_workers=args.w,    batch_size=args.b, shuffle=args.s)'
        , 'lsn': 5})
    cifar100_test_loader = get_test_dataloader(settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD, num_workers=args.w, batch_size=args.b,
        shuffle=args.s)
    Flog.flagged() and flog.write({'locals': [{'cifar100_test_loader': flog
        .serialize(cifar100_test_loader, 'cifar100_test_loader')}],
        'lineage':
        'cifar100_test_loader = get_test_dataloader(settings.CIFAR100_TRAIN_MEAN,    settings.CIFAR100_TRAIN_STD, num_workers=args.w, batch_size=args.b,    shuffle=args.s)'
        , 'lsn': 6})
    loss_function = nn.CrossEntropyLoss()
    Flog.flagged() and flog.write({'locals': [{'loss_function': flog.
        serialize(loss_function, 'loss_function')}], 'lineage':
        'loss_function = nn.CrossEntropyLoss()', 'lsn': 7})
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=0.0005)
    Flog.flagged() and flog.write({'locals': [{'optimizer': flog.serialize(
        optimizer, 'optimizer')}], 'lineage':
        'optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,    weight_decay=0.0005)'
        , 'lsn': 8})
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=
        settings.MILESTONES, gamma=0.2)
    Flog.flagged() and flog.write({'locals': [{'train_scheduler': flog.
        serialize(train_scheduler, 'train_scheduler')}], 'lineage':
        'train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=    settings.MILESTONES, gamma=0.2)'
        , 'lsn': 9})
    iter_per_epoch = len(cifar100_training_loader)
    Flog.flagged() and flog.write({'locals': [{'iter_per_epoch': flog.
        serialize(iter_per_epoch, 'iter_per_epoch')}], 'lineage':
        'iter_per_epoch = len(cifar100_training_loader)', 'lsn': 10})
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    Flog.flagged() and flog.write({'locals': [{'warmup_scheduler': flog.
        serialize(warmup_scheduler, 'warmup_scheduler')}], 'lineage':
        'warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)',
        'lsn': 11})
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net,
        settings.TIME_NOW)
    Flog.flagged() and flog.write({'locals': [{'checkpoint_path': flog.
        serialize(checkpoint_path, 'checkpoint_path')}], 'lineage':
        'checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings    .TIME_NOW)'
        , 'lsn': 12})
    if not os.path.exists(settings.LOG_DIR):
        Flog.flagged() and flog.write({'conditional_fork':
            '(not os.path.exists(settings.LOG_DIR))', 'lsn': 13})
        os.mkdir(settings.LOG_DIR)
    else:
        Flog.flagged() and flog.write({'conditional_fork':
            'not ((not os.path.exists(settings.LOG_DIR)))', 'lsn': 14})
    writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net,
        settings.TIME_NOW))
    Flog.flagged() and flog.write({'locals': [{'writer': flog.serialize(
        writer, 'writer')}], 'lineage':
        'writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net,    settings.TIME_NOW))'
        , 'lsn': 15})
    input_tensor = torch.Tensor(12, 3, 32, 32).cuda()
    Flog.flagged() and flog.write({'locals': [{'input_tensor': flog.
        serialize(input_tensor, 'input_tensor')}], 'lineage':
        'input_tensor = torch.Tensor(12, 3, 32, 32).cuda()', 'lsn': 16})
    writer.add_graph(net, Variable(input_tensor, requires_grad=True))
    if not os.path.exists(checkpoint_path):
        Flog.flagged() and flog.write({'conditional_fork':
            '(not os.path.exists(checkpoint_path))', 'lsn': 17})
        os.makedirs(checkpoint_path)
    else:
        Flog.flagged() and flog.write({'conditional_fork':
            'not ((not os.path.exists(checkpoint_path)))', 'lsn': 18})
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    Flog.flagged() and flog.write({'locals': [{'checkpoint_path': flog.
        serialize(checkpoint_path, 'checkpoint_path')}], 'lineage':
        'checkpoint_path = os.path.join(checkpoint_path, "{net}-{epoch}-{type}.pth")'
        , 'lsn': 19})
    best_acc = 0.0
    Flog.flagged() and flog.write({'locals': [{'best_acc': flog.serialize(
        best_acc, 'best_acc')}], 'lineage': 'best_acc = 0.0', 'lsn': 20})
    for epoch in range(1, settings.EPOCH):
        Flog.flagged() and flog.write({'start_loop': 157, 'lsn': 21})
        if epoch > args.warm:
            Flog.flagged() and flog.write({'conditional_fork':
                '(epoch > args.warm)', 'lsn': 23})
            train_scheduler.step(epoch)
        else:
            Flog.flagged() and flog.write({'conditional_fork':
                'not ((epoch > args.warm))', 'lsn': 24})
        train(epoch)
        acc = eval_training(epoch)
        Flog.flagged() and flog.write({'locals': [{'acc': flog.serialize(
            acc, 'acc')}], 'lineage': 'acc = eval_training(epoch)', 'lsn': 25})
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            Flog.flagged() and flog.write({'conditional_fork':
                '(epoch > settings.MILESTONES[1] and best_acc < acc)',
                'lsn': 26})
            torch.save(net.state_dict(), checkpoint_path.format(net=args.
                net, epoch=epoch, type='best'))
            best_acc = acc
            Flog.flagged() and flog.write({'locals': [{'best_acc': flog.
                serialize(best_acc, 'best_acc')}], 'lineage':
                'best_acc = acc', 'lsn': 28})
            continue
        else:
            Flog.flagged() and flog.write({'conditional_fork':
                'not ((epoch > settings.MILESTONES[1] and best_acc < acc))',
                'lsn': 27})
        if not epoch % settings.SAVE_EPOCH:
            Flog.flagged() and flog.write({'conditional_fork':
                '(not epoch % settings.SAVE_EPOCH)', 'lsn': 29})
            torch.save(net.state_dict(), checkpoint_path.format(net=args.
                net, epoch=epoch, type='regular'))
        else:
            Flog.flagged() and flog.write({'conditional_fork':
                'not ((not epoch % settings.SAVE_EPOCH))', 'lsn': 30})
        Flog.flagged() and flog.write({'end_loop': 157, 'lsn': 22})
    writer.close()
else:
    Flog.flagged() and flog.write({'conditional_fork':
        'not ((__name__ == "__main__"))', 'lsn': 1})
Flog.flagged() and flog.write({'end_of_file': 'train.py', 'lsn': 32})
Flog.flagged() and flog.writer.close()
try:
    del flog
except:
    pass