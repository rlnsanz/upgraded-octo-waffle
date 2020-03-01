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

def train(epoch):
    try:
        flor.namespace_stack.new()
        net.train()
        flor.skip_stack.new(0)
        if flor.skip_stack.peek().should_execute((not flor.SKIP)):
            for (batch_index, (images, labels)) in enumerate(imagenet_train_loader):
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
                fprint('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(loss.item(), optimizer.param_groups[0]['lr'], epoch=epoch, trained_samples=((batch_index * args.b) + len(images)), total_samples=len(imagenet_train_loader.dataset)))
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
            for (images, labels) in imagenet_val_loader:
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
        return ((test_loss / len(imagenet_val_loader.dataset)), (correct.float() / len(imagenet_val_loader.dataset)))
    finally:
        flor.namespace_stack.pop()

@ray.remote(num_cpus=4, num_gpus=1)
def do_partition(partition, device_id):
    # You have to set the right variables as global for visibility
    global net, optimizer, clr_scheduler, loss_function, fprint

    # The predecessor you load, all else is re-executed
    predecessor_epoch = partition[0] - 1

    if not flor.is_initialized():
        # Ray creates a new instance of the library per worker, so we have to re-init
        flor.initialize(**user_settings, predecessor_id=predecessor_epoch)

    # This line is so parallel workers don't collide
    fprint = flor.utils.fprint(['/home/ec2-user/', 'flor_output'], device_id)

    # Do the general initialization
    # The code below is copy/pasteed from __main__
    # Each worker needs to initialize its own Neural Net so it's in the right GPU
    # Anything that goes on the GPU or reads from the GPU has to be initialized in each worker
    net = get_network(args, use_gpu=True)
    flor.namespace_stack.test_force(net, 'net')
    loss_function = nn.CrossEntropyLoss()
    flor.namespace_stack.test_force(loss_function, 'loss_function')
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0, weight_decay=0.0)
    flor.namespace_stack.test_force(optimizer, 'optimizer')
    clr_scheduler = CLR_Scheduler(optimizer, net_steps=(iter_per_epoch * settings.EPOCH), min_lr=args.lr,
                                  max_lr=3.0, tail_frac=0.0)
    flor.namespace_stack.test_force(clr_scheduler, 'clr_scheduler')

    # Load the end state of the predecessor so we can re-execute in the middle
    if predecessor_epoch >= 0:
        # Initialize the Previous Epoch
        train(predecessor_epoch)
        eval_training(predecessor_epoch)

    # Re-execute in the middle
    flor.SKIP = False   # THIS IS IMPORTANT, otherwise flor will SKIP
    for epoch in partition:
        # This is just good old fashined re-execution
        train(epoch)
        (loss, acc) = eval_training(epoch)
        fprint('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(loss, acc))

    # Clear the memory for cleanliness, this step might be optional
    torch.cuda.empty_cache()

if (__name__ == '__main__'):
    # INITIALIZE RAY INSIDE __main__, have a redis password always
    ray.init(redis_password="pa-pa-password")

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    imagenet_train_loader = get_training_dataloader(
        None,
        None,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s,
        dataset='imagenet'
    )

    flor.namespace_stack.test_force(imagenet_train_loader, 'imagenet_train_loader')

    imagenet_val_loader = get_test_dataloader(
        None,
        None,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s,
        dataset='imagenet'
    )

    flor.namespace_stack.test_force(imagenet_val_loader, 'imagenet_val_loader')

    iter_per_epoch = len(imagenet_train_loader)

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    best_acc = 0.0

    # Into get_partitions, you will want to pass the iterator that you iterate over in the
    # Outermost loop. The second arg is the number of GPUs in your machine
    partitions = flor.utils.get_partitions(range(8), 4)


    user_settings = flor.user_settings # IMPORTANT: This is for re-initialization. Don't forget it
    # Ray stuff:
    futures = [do_partition.remote(p, i) for i,p in enumerate(partitions)]
    ray.get(futures)

    print('------- {} seconds ---------'.format((time.time() - start_time)))
