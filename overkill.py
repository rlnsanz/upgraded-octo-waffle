# train.py
#!/usr/bin/env  python3

import time
start_time = time.time()

import flor

import os
import sys
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, CLR_Scheduler, Dynamic_CLR_Scheduler

def train(epoch):
    try:
        flor.namespace_stack.new()
        net.train()
        flor.skip_stack.new(0)
        if flor.skip_stack.peek().should_execute((not flor.SKIP)):
            for (batch_index, (images, labels)) in enumerate(cifar100_training_loader):
                flor.skip_stack.new(1)
                if flor.skip_stack.peek().should_execute((not flor.SKIP)):
                    clr_scheduler.step()
                    images = Variable(images)
                    flor.namespace_stack.test_force(images, 'images')
                    labels = Variable(labels)
                    flor.namespace_stack.test_force(labels, 'labels')
                    if torch.cuda.is_available():
                        labels = labels.cuda()
                        flor.namespace_stack.test_force(labels, 'labels')
                        images = images.cuda()
                        flor.namespace_stack.test_force(images, 'images')
                    optimizer.zero_grad()
                    outputs = net(images)
                    flor.namespace_stack.test_force(outputs, 'outputs')
                    loss = loss_function(outputs, labels)
                    flor.namespace_stack.test_force(loss, 'loss')
                    loss.backward()
                    optimizer.step()
                    print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(loss.item(), optimizer.param_groups[0]['lr'], epoch=epoch, trained_samples=((batch_index * args.b) + len(images)), total_samples=len(cifar100_training_loader.dataset)))
                (_, _) = flor.skip_stack.pop().proc_side_effects(clr_scheduler, optimizer)
        (_, _) = flor.skip_stack.pop().proc_side_effects(clr_scheduler, optimizer)
    finally:
        flor.namespace_stack.pop()

def eval_training(epoch):
    try:
        flor.namespace_stack.new()
        net.eval()
        test_loss = 0.0
        flor.namespace_stack.test_force(test_loss, 'test_loss')
        correct = 0.0
        flor.namespace_stack.test_force(correct, 'correct')
        flor.skip_stack.new(2)
        if flor.skip_stack.peek().should_execute((not flor.SKIP)):
            for (images, labels) in cifar100_test_loader:
                flor.skip_stack.new(3)
                if flor.skip_stack.peek().should_execute((not flor.SKIP)):
                    images = Variable(images)
                    flor.namespace_stack.test_force(images, 'images')
                    labels = Variable(labels)
                    flor.namespace_stack.test_force(labels, 'labels')
                    if torch.cuda.is_available():
                        images = images.cuda()
                        flor.namespace_stack.test_force(images, 'images')
                        labels = labels.cuda()
                        flor.namespace_stack.test_force(labels, 'labels')
                    outputs = net(images)
                    flor.namespace_stack.test_force(outputs, 'outputs')
                    loss = loss_function(outputs, labels)
                    flor.namespace_stack.test_force(loss, 'loss')
                    test_loss += loss.item()
                    (_, preds) = outputs.max(1)
                    flor.namespace_stack.test_force(_, '_')
                    flor.namespace_stack.test_force(preds, 'preds')
                    correct += preds.eq(labels).sum()
                (test_loss, correct) = flor.skip_stack.pop().proc_side_effects(test_loss, correct)
        (test_loss, correct) = flor.skip_stack.pop().proc_side_effects(test_loss, correct)
        return ((test_loss / len(cifar100_test_loader.dataset)), (correct.float() / len(cifar100_test_loader.dataset)))
    finally:
        flor.namespace_stack.pop()

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    flor.namespace_stack.test_force(parser, 'parser')
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()
    flor.namespace_stack.test_force(args, 'args')
    net = get_network(args, use_gpu=args.gpu)
    flor.namespace_stack.test_force(net, 'net')
    cifar100_training_loader = get_training_dataloader(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD, num_workers=args.w, batch_size=args.b, shuffle=args.s)
    flor.namespace_stack.test_force(cifar100_training_loader, 'cifar100_training_loader')
    cifar100_test_loader = get_test_dataloader(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD, num_workers=args.w, batch_size=args.b, shuffle=args.s)
    flor.namespace_stack.test_force(cifar100_test_loader, 'cifar100_test_loader')
    iter_per_epoch = len(cifar100_training_loader)
    flor.namespace_stack.test_force(iter_per_epoch, 'iter_per_epoch')
    loss_function = nn.CrossEntropyLoss()
    flor.namespace_stack.test_force(loss_function, 'loss_function')
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0, weight_decay=0.0)
    flor.namespace_stack.test_force(optimizer, 'optimizer')
    clr_scheduler = CLR_Scheduler(optimizer, net_steps=(iter_per_epoch * settings.EPOCH), min_lr=args.lr, max_lr=3.0, tail_frac=0.0)
    flor.namespace_stack.test_force(clr_scheduler, 'clr_scheduler')
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    flor.namespace_stack.test_force(checkpoint_path, 'checkpoint_path')
    flor.namespace_stack.test_force(start_time, 'start_time')
    best_acc = 0.0
    flor.namespace_stack.test_force(best_acc, 'best_acc')
    epoch = 1
    flor.namespace_stack.test_force(epoch, 'epoch')
    for _ in range(settings.EPOCH):
        train(epoch)
        (loss, acc) = eval_training(epoch)
        print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(loss, acc))
        epoch += 1
    print('------- {} seconds ---------'.format((time.time() - start_time)))
    if not flor.SKIP:
        flor.flush()
