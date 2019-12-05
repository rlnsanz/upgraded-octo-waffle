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

import ray
ray.init()

def flor_writer(device_id):
    def write(s):
        with open("flor_output_{}.txt".format(device_id), 'a') as f:
            f.write(s + '\n')
    return write

def train(epoch):
    try:
        flor.namespace_stack.new()
        net.train()
        flor.skip_stack.new(0)
        if flor.skip_stack.peek().should_execute((not flor.SKIP)):
            for (batch_index, (images, labels)) in enumerate(cifar100_training_loader):
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
                fprint('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(loss.item(), optimizer.param_groups[0]['lr'], epoch=epoch, trained_samples=((batch_index * args.b) + len(images)), total_samples=len(cifar100_training_loader.dataset)))
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
        flor.skip_stack.new(1)
        if flor.skip_stack.peek().should_execute((not flor.SKIP)):
            for (images, labels) in cifar100_test_loader:
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
        return ((test_loss / len(cifar100_test_loader.dataset)), (correct.float() / len(cifar100_test_loader.dataset)))
    finally:
        flor.namespace_stack.pop()

@ray.remote(num_gpus=8)
def do_partition(partition, device_id):
    global net, optimizer, clr_scheduler, loss_function, fprint
    predecessors_epoch = partition[0] - 1
    fprint = flor_writer(device_id)

    with torch.cuda.device(device_id):
        # Do the general initialization
        net = get_network(args, use_gpu=args.gpu)
        flor.namespace_stack.test_force(net, 'net')
        loss_function = nn.CrossEntropyLoss()
        flor.namespace_stack.test_force(loss_function, 'loss_function')
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0, weight_decay=0.0)
        flor.namespace_stack.test_force(optimizer, 'optimizer')
        clr_scheduler = CLR_Scheduler(optimizer, net_steps=(iter_per_epoch * settings.EPOCH), min_lr=args.lr,
                                      max_lr=3.0, tail_frac=0.0)
        flor.namespace_stack.test_force(clr_scheduler, 'clr_scheduler')

        if predecessors_epoch >= 0:
            # Initialize the Previous Epoch
            flor.writer.Writer.store_load = flor.writer.Writer.partitioned_store_load[predecessors_epoch]
            train(predecessors_epoch)
            eval_training(predecessors_epoch)

        flor.SKIP = False
        for epoch in partition:
            train(epoch)
            (loss, acc) = eval_training(epoch)
            fprint('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(loss, acc))


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

    cifar100_training_loader = get_training_dataloader(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD, num_workers=args.w, batch_size=args.b, shuffle=args.s)
    cifar100_test_loader = get_test_dataloader(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD, num_workers=args.w, batch_size=args.b, shuffle=args.s)
    iter_per_epoch = len(cifar100_training_loader)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    best_acc = 0.0

    import math
    iterator = range(settings.EPOCH)
    NUM_GPU = 8
    work_per_gpu = math.ceil(len(iterator) / NUM_GPU)
    i = 0
    partitions = []
    while i * work_per_gpu < len(iterator):
        partitions.append(iterator[i*work_per_gpu: (i+1)*work_per_gpu])
        i += 1

    """
    In [27]: partitions
    Out[27]:
    [range(0, 3),
     range(3, 6),
     range(6, 9),
     range(9, 12),
     range(12, 15),
     range(15, 18),
     range(18, 20)]
    """

    futures = [do_partition.remote(p, i) for i,p in enumerate(partitions)]
    ray.get(futures)

    print('------- {} seconds ---------'.format((time.time() - start_time)))

