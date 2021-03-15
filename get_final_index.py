import numpy as np
from network import VGG
from parameter import get_parameter
import torch


class FinalIndex:
    def __init__(self, network, index, layer_idx):
        self.network = network
        self.index = index
        self.layer_idx = layer_idx

    def get_left_index(self, mask_ite, conv_ite):
        # mask = mask.tolist()
        for idx, k in enumerate(mask_ite):
            if idx in conv_ite:
                mask_ite[idx] = -1
        for item in mask_ite[::-1]:
            if item == -1:
                mask_ite.remove(item)
        return mask_ite

    def get_one_conv(self, mask, conv):
        mask = mask.tolist()
        index = 0
        while index < len(conv):
            mask = self.get_left_index(mask, conv[index])
            index += 1
        return mask

    def init_mask(self, prune_layers):
        conv_count = 0
        mask_all = []
        for i in range(len(self.network.features)):
            if isinstance(self.network.features[i], torch.nn.Conv2d):
                if 'conv%d' % conv_count in prune_layers:
                    mask_length_1 = self.get_size(self.network.features[i].bias.data)
                    mask_length = mask_length_1[0]
                    # print(type(mask_length))
                    # print(mask_length)
                    mask_conv = np.arange(mask_length)
                    mask_all.append(mask_conv)

                conv_count += 1

        return mask_all

    def get_size(self, conv):
        conv_size = conv.size()
        return conv_size

    def get_all_conv(self, mask_all):
        names = locals()
        # layer_idx = 13
        for i in range(self.layer_idx):
            names['conv' + str(i)] = []

        for key, value in self.index.items():
            for i in range(self.layer_idx):
                names.get('conv' + str(i)).append(value[i])

        mask_final = []
        for i in range(self.layer_idx):
            mask_oneconv_final = self.get_one_conv(mask_all[i], names.get('conv'+ str(i)))
            mask_final.append(mask_oneconv_final)
        return mask_final

    def get_pruned_index(self, mask_left, mask):
        mask = mask.tolist()
        for idx, k in enumerate(mask):
            if k in mask_left:
                mask[idx] = -1
        for item in mask[::-1]:
            if item == -1:
                mask.remove(item)
        return mask

    def get_final_pruned_index(self, mask_final, mask_all):
        feature = []
        for i in range(len(self.network.features)):
            if isinstance(self.network.features[i], torch.nn.Conv2d):
                feature.append(i)

        pruned_index = []
        for i in range(self.layer_idx):
            pruned_index_oneconv = self.get_pruned_index(mask_final[i], mask_all[i])
            pruned_index.append(pruned_index_oneconv)

        final_pruned_index = {}
        for i, val in enumerate(feature):
            final_pruned_index['feature.%d' % val] = pruned_index[i]
        return final_pruned_index
