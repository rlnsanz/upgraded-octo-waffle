# train.py
#!/usr/bin/env  python3

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

    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader): #batch_index, images, labels shadowed at end of loop

        clr_scheduler.step()                    # changes clr_scheduler
        images = Variable(images)               # changes images:out; not_changes Variable, images:in
        labels = Variable(labels)               # changes labels:out; not_changes Variable, labels:in

        if torch.cuda.is_available():           # not_changes torch,torch.cuda,torch.cuda.is_available
            labels = labels.cuda()              # changes labels:out, 
            images = images.cuda()              # changes images

        optimizer.zero_grad()                   # changes optimizer
        outputs = net(images)                   # changes outputs; not_changes net, images
        loss = loss_function(outputs, labels)   # changes loss; not_changes loss_function, outputs, labels
        loss.backward()                         # changes loss
        optimizer.step()                        # changes optimizer

        tblogger.small_step(loss, batch_index)

        if batch_index % 10 == 0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(loss.item(), optimizer.param_groups[0]['lr'], epoch=epoch, trained_samples=batch_index * args.b + len(images), total_samples=len(cifar100_training_loader.dataset)))                                      # Could have side-effects, and I can't analyze them, so I should replay it


def eval_training(epoch):
    net.eval()                                                      # changes net

    test_loss = 0.0 
    correct = 0.0   

    for (images, labels) in cifar100_test_loader:
        images = Variable(images)                                   # changes images
        labels = Variable(labels)                                   # changes labels

        if torch.cuda.is_available():
            images = images.cuda()                                  # changes images
            labels = labels.cuda()                                  # changes labels

        outputs = net(images)                                       # changes outputs
        loss = loss_function(outputs, labels)                       # changes loss
        test_loss += loss.item()                                    # changes test_loss
        _, preds = outputs.max(1)                                   # changes _, preds
        correct += preds.eq(labels).sum()                           # changes correct


    return test_loss / len(cifar100_test_loader.dataset), correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':
    args = get_args()

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

    iter_per_epoch = len(cifar100_training_loader)                                       
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0, weight_decay=0.0)         
    clr_scheduler = CLR_Scheduler(optimizer, net_steps=(iter_per_epoch * args.epoch), min_lr=args.lr, max_lr=3.0, tail_frac=0.0) #memoize?
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)                                                             

    best_acc = 0.0
    tblogger = TBLogger(args,  net, optimizer, start_epoch=0, iter_per_epoch=iter_per_epoch, eric=True)
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        train(epoch)                        #changes net,optimizer,clr_scheduler;not_changes train, epoch
        loss, acc = eval_training(epoch)    #changes loss, acc, net                                                  

        tblogger.big_step(loss, acc)
        print(f"------- {time.time() - epoch_start_time} segundos cada epoca --------- owner: {args.owner}")
        print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
            loss,
            acc
        ))
        if epoch in [25, 50, 75]:
            tblogger.flush(fork=True)

    print(f"------- {time.time() - start_time} seconds --------- owner: {args.owner}")
    tblogger.close()
    tblogger.flush(fork=True)

