""" helper function

author baiyu
"""

import sys

import numpy
import argparse

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from datetime import datetime
from conf import settings

class TBLogger:

    def __init__(self, args, net, optimizer, start_epoch, iter_per_epoch, eric=False):
        self.eric = eric
        owner = args.owner
        assert owner in ['judy', 'mike', 'chuck', 'flor']
        self.owner = owner
        self.writer = SummaryWriter(f'{owner}/{datetime.now().isoformat()}')
        self.args = args
        self.loglvl = int(args.loglvl)
        self.logfreq = int(args.logfreq)
        self.iter_per_epoch = iter_per_epoch

        self.epoch = start_epoch
        self.net = net
        self.optimizer = optimizer

        self.total_epochs = args.epoch

        self.buffer = []

        if self.owner == 'flor' and self.loglvl > 0:
            self.loglvl = 4

    def big_step(self, loss, acc):
        self.writer.add_scalar('metric/val_loss', loss, self.iter_per_epoch * (self.epoch + 1))
        self.writer.add_scalar('metric/accuracy', acc, self.iter_per_epoch * (self.epoch + 1))

        self.epoch += 1

    def small_step(self, loss, batch_index):
        self.writer.add_scalar('metric/loss', loss.item(), self.epoch*self.iter_per_epoch + batch_index)
        self.writer.add_scalar('param/lr', self.optimizer.param_groups[0]['lr'], self.epoch*self.iter_per_epoch + batch_index)

        if self.do(batch_index):
            print('heavy serializing')
            for k in self.net.activations:
                if not self.eric:
                    self.writer.add_histogram(f'activations/{k}', self.net.activations[k], self.epoch*self.iter_per_epoch + batch_index)
                else:
                    self.buffer.append((f'activations/{k}', self.net.activations[k].cpu(), self.epoch*self.iter_per_epoch + batch_index))

            for n, p in self.net.named_parameters():
                if not self.eric:
                    self.writer.add_histogram(f'weight/{n}', p, self.epoch*self.iter_per_epoch + batch_index)
                else:
                    self.buffer.append((f'weight/{n}', p.cpu(), self.epoch*self.iter_per_epoch + batch_index))
                if p.requires_grad:
                    if not self.eric:
                        self.writer.add_histogram(f'grad/{n}', p.grad, self.epoch*self.iter_per_epoch + batch_index)
                    else:
                        self.buffer.append((f'grad/{n}', p.grad.cpu(), self.epoch*self.iter_per_epoch + batch_index))

    def do(self, batch_index):
        work_epochs = int((self.total_epochs * self.loglvl) / 4)
        if work_epochs == 0:
            return False
        if self.epoch in range(self.total_epochs)[-1*work_epochs:]:
            return batch_index % self.logfreq == 0
        else:
            return False


    def close(self):
        self.writer.close()




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-loglvl', type=int, required=True, help='log level')
    parser.add_argument('-logfreq', type=int, default=100, help='log frequency')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-epoch', type=int, default=100, help='number of epochs for training')
    parser.add_argument('-owner', type=str, required=True, help='experiment owner')
    args = parser.parse_args()

    args.truloglvl = args.loglvl

    """
    All levels log loss and lr
    
    Activations, weights, and gradients conditionally logged:
    -1: 
    0: No heavy logging
    1: Log last quartile
    2: Log last half
    3: Log last three quartiles
    4: Log it all
    """
    assert args.loglvl in range(5)
    assert args.owner in ['judy', 'mike', 'chuck', 'flor']

    if args.owner == 'judy':
        assert args.loglvl > 0
        args.epoch = int((args.epoch * args.loglvl) / 4)
        args.loglvl = 4
    elif args.owner == 'chuck':
        pass
    elif args.owner == 'mike':
        args.loglvl = 4
    elif args.owner == 'flor':
        assert args.loglvl > 0
        import flor
        epoch = int((args.epoch * args.loglvl) / 4)
        args.epoch = len(flor.utils.get_partitions(epoch, 8, True, 1)[0])
        args.loglvl = 4

    return args

def get_network(args, use_gpu=True):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'squeezenetlog':
        from models.squeezenet import squeezenetlog
        net = squeezenetlog()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34 
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50 
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101 
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    
    if use_gpu and torch.cuda.is_available():
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data
    
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class CLR_Scheduler(_LRScheduler):
    def __init__(self, optimizer, net_steps, min_lr, max_lr, last_epoch=-1, repeat_factor=1, tail_frac=0.5):
        """
        Implemented for Super Convergence

        :param optimizer:
        :param net_steps: Number of calls to step() overall
        :param min_lr:
        :param max_lr:
        :param last_epoch:
        :param tail_frac: This scheduler consists of a cycle followed by a long tail that decreases monotonically.
            Tail frac is the fraction of net_steps allocated to the tail.
        """
        # The +1 is because get_lr is called in super().__init__
        tail_step_size = int(net_steps * tail_frac)
        step_size = int((net_steps - tail_step_size) / 2)
        self.lr_schedule = [min_lr,] + list(
            numpy.repeat(list(
                numpy.linspace(min_lr, max_lr, int(numpy.ceil(step_size / repeat_factor)), endpoint=False)) +
                         list(
                             numpy.linspace(max_lr, min_lr, int(numpy.floor(step_size / repeat_factor)))),
                         repeat_factor))
        tail_step_size = net_steps - len(self.lr_schedule) + 1
        self.lr_schedule += list(numpy.linspace(min_lr, min_lr/4, tail_step_size))
        assert len(self.lr_schedule) == net_steps + 1
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.lr_schedule.pop(0),]

    def loop_next(self, prev_accuracy):
        return len(self.lr_schedule) > 0


class Dynamic_CLR_Scheduler(_LRScheduler):
    def __init__(self, optimizer, epoch_per_cycle, iter_per_epoch, epoch_per_tail, min_lr, max_lr, target=0.8):
        self.step_size = (epoch_per_cycle * iter_per_epoch) / 2
        self.iter_per_epoch = iter_per_epoch
        self.epoch_per_tail = epoch_per_tail
        self.min_lr = min_lr
        self.max_lr = max_lr

        self.lr_schedule = [min_lr,] + self.get_lr_schedule()

        self.acc_mem = []
        self.target = target
        super().__init__(optimizer)

    def get_lr_schedule(self):
        lr_sched = list(numpy.linspace(self.min_lr, self.max_lr, int(self.step_size), endpoint=False)) +list(numpy.linspace(self.max_lr, self.min_lr, int(numpy.ceil(self.step_size))))
        lr_sched += list(numpy.linspace(self.min_lr, self.min_lr/2, self.epoch_per_tail * self.iter_per_epoch))
        return lr_sched


    def get_lr(self):
        return [self.lr_schedule.pop(0),]

    def loop_next(self, prev_accuracy):
        if prev_accuracy is None:
            return True
        if len(self.lr_schedule) > self.epoch_per_tail * self.iter_per_epoch:
            return True
        elif self.lr_schedule:
            self.acc_mem.append(prev_accuracy)
            return True
        else:
            # Options
            self.acc_mem.append(prev_accuracy)
            if any(filter(lambda x: x >= self.target, self.acc_mem)):
                return False
            else:
                epsilon = 3e-3
                # We're making progress and should continue our trend
                if numpy.array(list(map(lambda x: float(x), self.acc_mem))).std() > epsilon:
                    self.acc_mem = []
                    self.lr_schedule = list(numpy.linspace(self.min_lr, self.min_lr/2, self.epoch_per_tail * self.iter_per_epoch))
                # We're not making progress and should jolt
                else:
                    self.acc_mem = []
                    self.lr_schedule = self.get_lr_schedule()
                return True





