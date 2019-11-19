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
from torch.autograd import Variable

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR

import cProfile

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

        if epoch % settings.LOG_STEPSIZE == 0:
            flor.store(test_loss)
            flor.store(correct)
    else:
        if epoch % settings.LOG_STEPSIZE == 0:
            test_loss = flor.load()
            correct = flor.load()
        else:
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

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset)
    ))
    print()

    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-net2', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    net, net2 = get_network(args, use_gpu=args.gpu)
        
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
    first_net = net 
    
    ### Training and evaluation of resnet18 extended ###
    net = net2
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net2, settings.TIME_NOW)

    best_acc = 0.0
    start_time = time.time()
    for epoch in range(1, settings.EPOCH):

        # if epoch > args.warm:
        #     train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)
    print("------- {} seconds ---------".format(time.time() - start_time))
    second_net = net

    ### Retrieve the number of weights in each of the corresponding filters ###
    first_conv1_shape = first_net.conv1[0].weight.flatten().shape[0] 
    first_conv2_shape = first_net.conv2_x[0].residual_function[3].weight.flatten().shape[0]
    first_conv3_shape = first_net.conv3_x[0].residual_function[3].weight.flatten().shape[0]
    first_conv4_shape = first_net.conv4_x[0].residual_function[3].weight.flatten().shape[0]
    first_conv5_shape = first_net.conv5_x[0].residual_function[3].weight.flatten().shape[0]
    first_fc_shape = first_net.fc.weight.flatten().shape[0]

    second_conv1_shape = second_net.conv1[0].weight.flatten().shape[0]
    second_conv2_shape = second_net.conv2_x[0].residual_function[3].weight.flatten().shape[0]
    second_conv3_shape = second_net.conv3_x[0].residual_function[3].weight.flatten().shape[0]
    second_conv4_shape = second_net.conv4_x[0].residual_function[3].weight.flatten().shape[0]
    second_conv5_shape = second_net.conv5_x[0].residual_function[3].weight.flatten().shape[0]
    second_fc_shape = second_net.fc.weight.flatten().shape[0]
    
    ### Extend the original resnet18 weight tensor to the length of the extended resnet18 weight tensor ###
    conv1_zeros = torch.zeros([1, second_conv1_shape-first_conv1_shape], dtype=torch.float32)[0]
    conv2_zeros = torch.zeros([1, second_conv2_shape-first_conv2_shape], dtype=torch.float32)[0]
    conv3_zeros = torch.zeros([1, second_conv3_shape-first_conv3_shape], dtype=torch.float32)[0]
    conv4_zeros = torch.zeros([1, second_conv4_shape-first_conv4_shape], dtype=torch.float32)[0]
    conv5_zeros = torch.zeros([1, second_conv5_shape-first_conv5_shape], dtype=torch.float32)[0]
    fc_zeros = torch.zeros([1, second_fc_shape-first_fc_shape], dtype=torch.float32)[0]

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    conv1_zeros = conv1_zeros.to(device)
    conv2_zeros = conv2_zeros.to(device)
    conv3_zeros = conv3_zeros.to(device)
    conv4_zeros = conv4_zeros.to(device)
    conv5_zeros = conv5_zeros.to(device)
    fc_zeros = fc_zeros.to(device)

    first_net_conv1_weights = torch.cat((first_net.conv1[0].weight.flatten(), conv1_zeros))
    first_net_conv2_weights = torch.cat((first_net.conv2_x[0].residual_function[3].weight.flatten(), conv2_zeros))
    first_net_conv3_weights = torch.cat((first_net.conv3_x[0].residual_function[3].weight.flatten(), conv3_zeros))
    first_net_conv4_weights = torch.cat((first_net.conv4_x[0].residual_function[3].weight.flatten(), conv4_zeros))
    first_net_conv5_weights = torch.cat((first_net.conv5_x[0].residual_function[3].weight.flatten(), conv5_zeros))
    first_net_fc_weights = torch.cat((first_net.fc.weight.flatten(), fc_zeros))

    second_net_conv1_weights = second_net.conv1[0].weight.flatten()
    second_net_conv2_weights = second_net.conv2_x[0].residual_function[3].weight.flatten()
    second_net_conv3_weights = second_net.conv3_x[0].residual_function[3].weight.flatten()
    second_net_conv4_weights = second_net.conv4_x[0].residual_function[3].weight.flatten()
    second_net_conv5_weights = second_net.conv5_x[0].residual_function[3].weight.flatten()
    second_net_fc_weights = second_net.fc.weight.flatten()

    ### Take the dot product between the corresponding filters ###
    conv1_dot_num = torch.dot(first_net_conv1_weights, second_net_conv1_weights)
    conv2_dot_num = torch.dot(first_net_conv2_weights, second_net_conv2_weights)
    conv3_dot_num = torch.dot(first_net_conv3_weights, second_net_conv3_weights)
    conv4_dot_num = torch.dot(first_net_conv4_weights, second_net_conv4_weights)
    conv5_dot_num = torch.dot(first_net_conv5_weights, second_net_conv5_weights)
    fc_dot_num = torch.dot(first_net_fc_weights, second_net_fc_weights)

    conv1_dot_denom = torch.norm(first_net_conv1_weights.view(1,-1)) * torch.norm(second_net_conv1_weights.view(1,-1))
    conv2_dot_denom = torch.norm(first_net_conv2_weights.view(1,-1)) * torch.norm(second_net_conv2_weights.view(1,-1))
    conv3_dot_denom = torch.norm(first_net_conv3_weights.view(1,-1)) * torch.norm(second_net_conv3_weights.view(1,-1))
    conv4_dot_denom = torch.norm(first_net_conv4_weights.view(1,-1)) * torch.norm(second_net_conv4_weights.view(1,-1))
    conv5_dot_denom = torch.norm(first_net_conv5_weights.view(1,-1)) * torch.norm(second_net_conv5_weights.view(1,-1))
    fc_dot_denom = torch.norm(first_net_fc_weights.view(1,-1)) * torch.norm(second_net_fc_weights.view(1,-1))

    conv1_dot = conv1_dot_num/conv1_dot_denom
    conv2_dot = conv2_dot_num/conv2_dot_denom
    conv3_dot = conv3_dot_num/conv3_dot_denom
    conv4_dot = conv4_dot_num/conv4_dot_denom
    conv5_dot = conv5_dot_num/conv5_dot_denom
    fc_dot = fc_dot_num/fc_dot_denom

    print("conv1 dot = " + str(conv1_dot))
    print("conv2 dot = " + str(conv2_dot))
    print("conv3 dot = " + str(conv3_dot))
    print("conv4 dot = " + str(conv4_dot))
    print("conv5 dot = " + str(conv5_dot))
    print("fc dot = " + str(fc_dot))

 # https://pytorch.org/tutorials/beginner/saving_loading_models.html
