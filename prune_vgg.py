import torch
from network import VGG
from train import train_step
import numpy as np
import re
from collections import Counter
from evaluate import test_network
from utils import get_data_set
from loss import Loss_Calculator
from optimizer import get_optimizer


def prune_network(args, network=None):
    device = torch.device("cuda" if args.gpu_no >= 0 else "cpu")

    # if network is None:
    #     if args.network == 'vgg16_bn':
    #         network = VGG(args.network, args.data_set)
    #     if args.load_path:
    #         check_point = torch.load(args.load_path, map_location=lambda storage, loc: storage)
    #         network.load_state_dict(check_point['state_dict'])
    # else:
    #     network = torch.load(args.load_path)

    print(network)
    print("\nBefore prune network test:\n")
    test_network(args, network=network)

    flops_baseline = com_FLOP(args, network)
    print("\nBaseline flops:\n")
    print(flops_baseline)
    params_baseline = get_parameters_vgg(args, network)
    print("\nBaseline params:\n")
    print(params_baseline)

    flops_ratio = 0
    pruned_index_all = {}
    idx = 0
    while flops_ratio <= args.flops_ratio:
        print("-*-" * 10 + "\n\tPrune network\n" + "-*-" * 10)
        network, pruned_index_network = prune_step(args, network)
        pruned_index_all['iterate%d' % idx] = pruned_index_network
        idx += 1

        print("\nPruned network:\n")
        print(network)

        # network = network.to(device)

        print("\nAfter prune network test:\n")
        test_network(args, network=network)

        flops_network = com_FLOP(args, network)
        print("\nThe FLOPs of Pruned Networks\n")
        print(flops_network)
        print("\nFlops_ratio:\n")
        flops_ratio = get_flop_ratio(flops_baseline, flops_network)
        print(flops_ratio)

        params_network = get_parameters_vgg(args, network)
        print("\nThe Params of Pruned Networks\n")
        print(params_network)
        print("\nParams_ratio:\n")
        params_ratio = get_param_ratio(params_baseline, params_network)
        print(params_ratio)

        if flops_ratio <= args.flops_ratio:
            # update arguments for retraining pruned network
            args.epoch = args.update_param_epoch
            print("-*-" * 10 + "\n\tUpdate Params\n" + "-*-" * 10)
        else:
            args.epoch = args.finetune_epoch
            print("-*-" * 10 + "\n\tFinetune network\n" + "-*-" * 10)

        args.lr = args.retrain_lr
        args.lr_milestone = None

        data_set = get_data_set(args, train_flag=True)

        loss_calculator = Loss_Calculator()

        optimizer, scheduler = get_optimizer(network, args)

        for epoch in range(args.start_epoch, args.epoch):

            network = network.to(device)

            # make shuffled data loader
            data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True)

            # train one epoch
            train_step(network, data_loader, loss_calculator, optimizer, device, epoch, args.print_freq)

            # adjust learning rate
            if scheduler is not None:
                scheduler.step()

            test_network(args, network=network)
        # if flops_ratio <= args.flops_ratio:
        #     torch.save(network, args.save_path + 'model_prune_vgg%.2f.pkl' % flops_ratio)
        # else:
        #     torch.save(network, args.save_path + 'model_prune_vgg_final.pkl')

    return network, pruned_index_all


def prune_step(args, network):
    network = network.cpu()
    channel_index_network = []
    conv_count = 1  # conv count for 'indexing_prune_layers'
    dim = 0  # 0: prune corresponding dim of filter weight [out_ch, in_ch, k1, k2]
    for i in range(len(network.features)):
        if isinstance(network.features[i], torch.nn.Conv2d):

            if dim == 1:
                new_, residue = get_new_conv(network.features[i], dim, channel_index)
                network.features[i] = new_
                dim ^= 1

            if 'conv%d' % conv_count in args.prune_layers:
                print("\n\tPrune %d\n" % i)
                channel_index = get_channel_index(args, network.features[i].weight.data)
                channel_index_network.append(channel_index)
                print(channel_index)
                channel_prune_ratio = len(channel_index) / network.features[i].weight.size(0)
                print("%d layer channel_prune_ratio is:%f " % (i, channel_prune_ratio))
                new_ = get_new_conv(network.features[i], dim, channel_index)
                network.features[i] = new_
                dim ^= 1

            else:
                residue = None
            conv_count += 1

        elif dim == 1 and isinstance(network.features[i], torch.nn.BatchNorm2d):
            new_ = get_new_norm(network.features[i], channel_index)
            network.features[i] = new_
    # update to check last conv layer pruned

    if 'conv13' in args.prune_layers:
        network.classifier[0] = get_new_linear(network.classifier[0], channel_index)

    return network, channel_index_network


def com_FLOP(args, network):

    size_feature_vgg16 = [32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2]
    conv_flop = 0
    bn_flop = 0

    index = 0

    if args.data_set == 'CIFAR10':
        numclasses = 10
    if args.data_set == 'CIFAR100':
        numclasses = 100

    for i in range(len(network.features)):
        if isinstance(network.features[i], torch.nn.Conv2d):
            c_out, c_in, k = get_conv_size(network.features[i].weight.data)

            conv_flop += (c_in * k * k + 1) * c_out * (size_feature_vgg16[index] ** 2)

            if isinstance(network.features[i + 1], torch.nn.BatchNorm2d):
                bn_size = get_bn_size(network.features[i + 1].weight.data)
                bn_flop += bn_size * (size_feature_vgg16[index] ** 2) * 2

            index += 1

    relu_flop = bn_flop / 2
    fc_flop = 512*512 + 512 * numclasses
    flop = conv_flop + bn_flop + relu_flop + fc_flop

    return flop


def get_parameters_vgg(args, network):

    if args.data_set == 'CIFAR10':
        num_classes = 10
    if args.data_set == 'CIFAR100':
        num_classes = 100
    conv_param = 0
    bn_param = 0
    index = 0
    for i in range(len(network.features)):
        if isinstance(network.features[i], torch.nn.Conv2d):
            c_out, c_in, k = get_conv_size(network.features[i].weight.data)

            conv_param += (c_in * k * k + 1) * c_out

            if isinstance(network.features[i + 1], torch.nn.BatchNorm2d):
                bn_size = get_bn_size(network.features[i + 1].weight.data)
                bn_param += bn_size * 2

            index += 1

    fc_param = 512*512 + 512*2 + 512 * num_classes
    parameters = conv_param + bn_param + fc_param
    return parameters


def get_conv_size(conv):
    return conv.size()[0], conv.size()[1], conv.size()[2]


def get_bn_size(bn):
    return bn.size()[0]


def get_flop_ratio(flops_baseline, flops_network):
    return 1 - flops_network/flops_baseline


def get_param_ratio(params_baseline, params_network):
    return 1 - params_network/params_baseline


def get_channel_index(args, kernel):
    new_kernel = kernel.view(kernel.size(0), -1)
    new_kernel = np.array(new_kernel)

    dis = dict()
    for j in range(new_kernel.shape[0] - 1):
        for k in range(j + 1, new_kernel.shape[0]):
            one_dis = np.sqrt(np.sum(np.square(new_kernel[j] - new_kernel[k])))
            dis.setdefault("%d,%d" % (j, k), one_dis)

    number = []
    for key, value in dis.items():
        number.append(value)
    number = np.mat(number)

    number_mean = np.mean(number)
    number_std = np.std(number, ddof=1)

    list_dis = []
    for key, value in dis.items():
        if value < (number_mean - number_std):
            list_dis.append(key)

    list_dis = ",".join(list_dis)
    num_list_new = re.findall(r"\d+", list_dis)
    num_list_new = list(map(int, num_list_new))

    result = Counter(num_list_new)

    kernel_prune_index = []

    for key, value in result.items():
        if value > args.ratio * new_kernel.shape[0]:
            kernel_prune_index.append(key)
    return kernel_prune_index


def index_remove(tensor, dim, index, removed=False):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    size_ = list(tensor.size())
    new_size = tensor.size(dim) - len(index)
    size_[dim] = new_size
    new_size = size_

    select_index = list(set(range(tensor.size(dim))) - set(index))
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))

    if removed:
        return new_tensor, torch.index_select(tensor, dim, torch.tensor(index))

    return new_tensor


def get_new_conv(conv, dim, channel_index):
    if dim == 0:
        new_conv = torch.nn.Conv2d(in_channels=conv.in_channels,
                                   out_channels=int(conv.out_channels - len(channel_index)),
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)

        new_conv.weight.data = index_remove(conv.weight.data, dim, channel_index)
        new_conv.bias.data = index_remove(conv.bias.data, dim, channel_index)

        return new_conv

    elif dim == 1:
        new_conv = torch.nn.Conv2d(in_channels=int(conv.in_channels - len(channel_index)),
                                   out_channels=conv.out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)

        new_weight = index_remove(conv.weight.data, dim, channel_index)
        residue = None
        # if independent_prune_flag:
        #    new_weight, residue = new_weight
        new_conv.weight.data = new_weight
        new_conv.bias.data = conv.bias.data

        return new_conv, residue


def get_new_norm(norm, channel_index):
    new_norm = torch.nn.BatchNorm2d(num_features=int(norm.num_features - len(channel_index)),
                                    eps=norm.eps,
                                    momentum=norm.momentum,
                                    affine=norm.affine,
                                    track_running_stats=norm.track_running_stats)

    new_norm.weight.data = index_remove(norm.weight.data, 0, channel_index)
    new_norm.bias.data = index_remove(norm.bias.data, 0, channel_index)

    if norm.track_running_stats:
        print(index_remove(norm.running_mean.data, 0, channel_index))
        print(new_norm.running_mean.data)
        new_norm.running_mean.data = index_remove(norm.running_mean.data, 0, channel_index)
        new_norm.running_var.data = index_remove(norm.running_var.data, 0, channel_index)

    return new_norm


def get_new_linear(linear, channel_index):
    new_linear = torch.nn.Linear(in_features=int(linear.in_features - len(channel_index)),
                                 out_features=linear.out_features,
                                 bias=linear.bias is not None)
    new_linear.weight.data = index_remove(linear.weight.data, 1, channel_index)
    new_linear.bias.data = linear.bias.data

    return new_linear




