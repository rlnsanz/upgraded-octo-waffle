# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
from datetime import datetime
import time

sys.path.append('../flor/')
import flor

import numpy as np
flor.pin_state(np)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR

def train(epoch):
    """
    :globals: net, cifar100_training_loader, args, warmup_scheduler, torch
              optimizer, loss_function
    :param epoch:
    :return:
    """

    if not flor.SKIP:
        net.train()
        for batch_index, (images, labels) in enumerate(cifar100_training_loader):
            # if epoch <= args.warm:
            #     warmup_scheduler.step()

            images = Variable(images)
            labels = Variable(labels)

            if torch.cuda.is_available():
                labels = labels.cuda()
                images = images.cuda()

            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(cifar100_training_loader.dataset)
            ))


        if epoch % settings.LOG_STEPSIZE == 0:  
            # Store the globals
            flor.store(net.state_dict())
            # if epoch <= args.warm:
                # flor.store(warmup_scheduler.state_dict())
            flor.store(optimizer.state_dict())
    else:
        if epoch % settings.LOG_STEPSIZE == 0:
            net.load_state_dict(flor.load())
            # if epoch <= args.warm:
            #     warmup_scheduler.load_state_dict(flor.load)
            optimizer.load_state_dict(flor.load())
        else:
            net.train()
            for batch_index, (images, labels) in enumerate(cifar100_training_loader):
                # if epoch <= args.warm:
                #     warmup_scheduler.step()

                images = Variable(images)
                labels = Variable(labels)

                if torch.cuda.is_available():
                    labels = labels.cuda()
                    images = images.cuda()

                optimizer.zero_grad()
                outputs = net(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                    loss.item(),
                    optimizer.param_groups[0]['lr'],
                    epoch=epoch,
                    trained_samples=batch_index * args.b + len(images),
                    total_samples=len(cifar100_training_loader.dataset)
            ))

def eval_training(epoch):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    if not flor.SKIP:
        for (images, labels) in cifar100_test_loader:
            images = Variable(images)
            labels = Variable(labels)

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            outputs = net(images)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()
        flor.store(test_loss)
        flor.store(correct)
    else:
        test_loss = flor.load()
        correct = flor.load()


    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset)
    ))
    print()

    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    net = get_network(args, use_gpu=args.gpu)
        
    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    best_acc = 0.0
    start_time = time.time()
    for epoch in range(1, settings.EPOCH):

        # if epoch > args.warm:
        #     train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)
    print("------- {} seconds ---------".format(time.time() - start_time))

 # https://pytorch.org/tutorials/beginner/saving_loading_models.html
