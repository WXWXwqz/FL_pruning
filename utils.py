import torch
import torchvision
import torchvision.transforms as transforms
import time
import sys
import os
import re

class AverageMeter(object):    
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_network(args):
    network = None
    if args.load_path:
        check_point = torch.load(args.load_path)
        if args.network == 'vgg':
            network = VGG(args.vgg, args.data_set)
            # if args.load_path:
            check_point = torch.load(args.load_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(check_point['state_dict'])
        elif args.network == 'resnet':
            network = resnet32()
            check_point = torch.load("./trained_models/resnet32.pth", map_location=lambda storage, loc: storage)
            network.load_state_dict(check_point['state_dict'])
    return network


def get_normalizer(data_set, inverse=False):
    if data_set == 'CIFAR10':
        MEAN = (0.4914, 0.4822, 0.4465)
        STD = (0.2023, 0.1994, 0.2010)

    elif data_set == 'CIFAR100':
        MEAN = (0.5071, 0.4867, 0.4408)
        STD = (0.2675, 0.2565, 0.2761)

    else:
        raise RuntimeError("Not expected data flag !!!")

    if inverse:
        MEAN = [-mean/std for mean, std in zip(MEAN, STD)]
        STD = [1/std for std in STD]

    return transforms.Normalize(MEAN, STD)


def get_transformer(data_set, imsize=None, cropsize=None, crop_padding=None, hflip=None):
    transformers = [] 
    if imsize:
        transformers.append(transforms.Resize(imsize))
    if cropsize:
        ## https://github.com/kuangliu/pytorch-cifar
        transformers.append(transforms.RandomCrop(cropsize, padding=crop_padding))
    if hflip:
        transformers.append(transforms.RandomHorizontalFlip(hflip))

    transformers.append(transforms.ToTensor())
    transformers.append(get_normalizer(data_set))
    
    return transforms.Compose(transformers)


def get_data_set(args, train_flag=True):
    if train_flag:
        data_set = torchvision.datasets.__dict__[args.data_set](root=args.data_path, train=True, 
                                       transform=get_transformer(args.data_set, args.imsize,
                                           args.cropsize, args.crop_padding, args.hflip), download=True)
    else:
        data_set = torchvision.datasets.__dict__[args.data_set](root=args.data_path, train=False, 
                                           transform=get_transformer(args.data_set), download=True)    
    return data_set


def make_print_to_file(path='./'):
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass
    fileName = 'log'
    sys.stdout = Logger(fileName + '.log', path=path)


