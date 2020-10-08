import flor
import time
start_time = time.time()
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

from utils import TBLogger, get_args

def train(epoch):
    try:
        flor.namespace_stack.new()
        net.train()
        flor.skip_stack.new(0)
        if flor.skip_stack.peek().should_execute(not flor.SKIP):
            for batch_index, (images, labels) in enumerate(
                cifar100_training_loader):
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
                tblogger.small_step(loss, batch_index)
                if batch_index % 10 == 0:
                    print(
                        'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'
                        .format(loss.item(), optimizer.param_groups[0]['lr'],
                        epoch=epoch, trained_samples=batch_index * args.b + len
                        (images), total_samples=len(cifar100_training_loader.
                        dataset)))
        _, _, _ = flor.skip_stack.pop().proc_side_effects(clr_scheduler,
            torch.cuda, optimizer)
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
        flor.skip_stack.new(1)
        if flor.skip_stack.peek().should_execute(False):
            for images, labels in cifar100_test_loader:
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
                _, preds = outputs.max(1)
                flor.namespace_stack.test_force(_, '_')
                flor.namespace_stack.test_force(preds, 'preds')
                correct += preds.eq(labels).sum()
        test_loss, correct, _ = flor.skip_stack.pop().proc_side_effects(
            test_loss, correct, torch.cuda)
        return test_loss / len(cifar100_test_loader.dataset), correct.float(
            ) / len(cifar100_test_loader.dataset)
    finally:
        flor.namespace_stack.pop()

temp_m = {
    1 : 13,
    2 : 7,
    3: 5,
    4: 4
}

if __name__ == '__main__':
    args = get_args()
    net = get_network(args, use_gpu=args.gpu)
    flor.namespace_stack.test_force(net, 'net')
    cifar100_training_loader = get_training_dataloader(settings.
        CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD, num_workers=args.
        w, batch_size=args.b, shuffle=args.s)
    flor.namespace_stack.test_force(cifar100_training_loader,
        'cifar100_training_loader')
    cifar100_test_loader = get_test_dataloader(settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD, num_workers=args.w, batch_size=args.b,
        shuffle=args.s)
    flor.namespace_stack.test_force(cifar100_test_loader,
        'cifar100_test_loader')
    iter_per_epoch = len(cifar100_training_loader)
    flor.namespace_stack.test_force(iter_per_epoch, 'iter_per_epoch')
    loss_function = nn.CrossEntropyLoss()
    flor.namespace_stack.test_force(loss_function, 'loss_function')
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0,
        weight_decay=0.0)
    flor.namespace_stack.test_force(optimizer, 'optimizer')
    clr_scheduler = CLR_Scheduler(optimizer, net_steps=iter_per_epoch *
        args.epoch, min_lr=args.lr, max_lr=3.0, tail_frac=0.0)
    flor.namespace_stack.test_force(clr_scheduler, 'clr_scheduler')
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net,
        settings.TIME_NOW)
    flor.namespace_stack.test_force(checkpoint_path, 'checkpoint_path')
    best_acc = 0.0
    flor.namespace_stack.test_force(best_acc, 'best_acc')

    tblogger = TBLogger(args, net, optimizer, start_epoch=0, iter_per_epoch=iter_per_epoch)
    flor.skip_stack.new(2, 0)
    for epoch in flor.partition(range(50), 1, temp_m[args.truloglvl]):
        train(epoch)
        loss, acc = eval_training(epoch)
        flor.namespace_stack.test_force(loss, 'loss')
        flor.namespace_stack.test_force(acc, 'acc')
        tblogger.big_step(loss, acc)
        print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
            loss, acc))
    flor.skip_stack.pop()
    print('------- {} seconds ---------'.format(time.time() - start_time))
    if not flor.SKIP:
        flor.flush()
    tblogger.close()
