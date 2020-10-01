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

import cloudpickle

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

        # LOG ACTIVATIONS
        for k in net.activations:
            my_logger.write(f"{k} -- {torch.norm(net.activations[k])}\n")

        for i, p in enumerate(net.parameters()):
            # LOG WEIGHTS
            my_logger.write(f"{i} -- {p.size()} -- {cloudpickle.dumps(p)}\n")
            # LOG GRADIENTS
            if p.requires_grad:
                my_logger.write(f"{i} -- {p.grad.size()} -- {cloudpickle.dumps(p.grad)}\n")

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(loss.item(), optimizer.param_groups[0]['lr'], epoch=epoch, trained_samples=batch_index * args.b + len(images), total_samples=len(cifar100_training_loader.dataset)))                                      # Could have side-effects, and I can't analyze them, so I should replay it
    my_logger.flush()

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
    parser = argparse.ArgumentParser()                                                         
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')           
    args = parser.parse_args()                                                                 

    assert 'log' in args.net
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
    clr_scheduler = CLR_Scheduler(optimizer, net_steps=(iter_per_epoch * settings.EPOCH), min_lr=args.lr, max_lr=3.0, tail_frac=0.0) #memoize?
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    os.makedirs(checkpoint_path)

    with open(os.path.join(checkpoint_path, 'log.txt'), 'w') as my_logger:
        best_acc = 0.0
        for epoch in range(settings.EPOCH):
            train(epoch)                        #changes net,optimizer,clr_scheduler;not_changes train, epoch
            torch.save(net.state_dict(), os.path.join(checkpoint_path, f'net_{epoch}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, f'opt_{epoch}.pt'))
            torch.save(clr_scheduler.state_dict(), os.path.join(checkpoint_path, f'clrsched_{epoch}.pt'))
            loss, acc = eval_training(epoch)    #changes loss, acc, net

            print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
                loss,
                acc
            ))

        print("------- {} seconds ---------".format(time.time() - start_time))
