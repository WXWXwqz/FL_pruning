from network import VGG
from parameter import get_parameter
from prune_vgg import prune_network
from get_final_index import FinalIndex

if __name__ == '__main__':
    args = get_parameter()

    network = None
    layer_idx = 13

    args.prune_layers = ['conv0', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'conv9', 'conv10', 'conv11',
                         'conv12', 'conv13']
    if args.prune_flag:
        network, prune_index = prune_network(args, network=None)

    old_model = VGG(args.network, args.data_set)
    
    final_index = FinalIndex(old_model, prune_index, layer_idx)
    final_index.layer_pruned_index()
    mask_all = final_index.init_mask(args)
    mask_final = final_index.get_all_conv(mask_all)
    final_pruned_index = final_index.get_final_pruned_index(mask_final, mask_all)




