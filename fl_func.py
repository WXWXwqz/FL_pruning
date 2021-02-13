import threading
import time
import numpy as np
import torch as th
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch.optim as optim 
from syft.frameworks.torch.fl import utils
import sys
from typing import List
import socket
import copy
import prune_vgg
from parameter import get_parameter
from get_final_index import FinalIndex
from network import VGG

class LetNet5(nn.Module):  # for MNIST
    def __init__(self, num_clases=10):
        super(LetNet5, self).__init__()
 
        self.c1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
 
        self.c2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
 
        self.c3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5),
            # nn.BatchNorm2d(120),
            nn.ReLU()
        )
 
        self.fc1 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
 
        self.fc2 = nn.Sequential(
            nn.Linear(84, 10),
            
         
        )
 
    def forward(self, x):
        out = self.c1(x)
        out = self.c2(out)
        out = self.c3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        
        return out

def model_para_print(model):

    params=model.named_parameters()
    for name,p in params:   
        print(name)
        print(p.shape)
        if len(p.shape)>1:
            tmp=torch.zeros(p.shape[0],p.shape[1])
            for i in range(p.shape[0]):
                for j in range(p.shape[1]):
                # print(i)
                # print(p[i])
                    if p[i][j][0][0]==0:
                        tmp[i][j]=0
                    else:
                        tmp[i][j]=1
                    print(p[i][j])
                    print(int(tmp[i][j].item()),end=' ')
                print('')
            # print(tmp)
        else:
            print(p)
def model_para_print2(model,model2):

    params=model.named_parameters()
    params2=model2.named_parameters()
    for name,p in params:   
        print(name)
        print(p.shape)
        if len(p.shape)>1:
            tmp=torch.zeros(p.shape[0],p.shape[1])
            for i in range(p.shape[0]):
                for j in range(p.shape[1]):
                # print(i)
                # print(p[i])
                    if p[i][j][0][0]==0:
                        tmp[i][j]=0
                    else:
                        tmp[i][j]=1
                        
                    print(p[i][j])
                    print(int(tmp[i][j].item()),end=' ')
                print('')
            # print(tmp)
        else:
            print(p)
def model_para_purn_compare_print(model1,model2):

    
    params1=model1.named_parameters()
    dict_dst_params1 = dict(params1)  
    params2=model2.named_parameters()
    dict_dst_params2 = dict(params2)  
    # print(type(params2))
    for name in dict_dst_params1:       
        print(name)
        if name in dict_dst_params2:
            print("origin",dict_dst_params1[name],type(dict_dst_params1[name]))
            print("pruned",dict_dst_params2[name],type(dict_dst_params2[name]))
            print("*-*--**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
        else:
            print("origin",dict_dst_params1[name].shape)
            print("pruned","all delete")
def model_para_clear(dst_model):  
    dst_params=dst_model.named_parameters()
    dict_dst_params = dict(dst_params)    
    for name in dict_dst_params:
        with torch.no_grad():            
            dict_dst_params[name].set_(torch.zeros(dict_dst_params[name].shape))
    return dst_model   
def model_para_set_num(dst_model,num):  
    dst_params=dst_model.named_parameters()
    dict_dst_params = dict(dst_params)    
    for name in dict_dst_params:
        with torch.no_grad():            
            dict_dst_params[name].set_(num+torch.zeros(dict_dst_params[name].shape))
            # dict_dst_params[name].set_(dict_dst_params[name])
    return dst_model  
def get_model_conv_name(model):
    model_dict=dict(model.named_parameters())
    conv_name_list=[]
    for name in model_dict:
        model_w=model_dict[name]
        if len(model_w.shape)>1 and ('features' in name):
            conv_name_list.append(name)
    return conv_name_list
def get_aved_conv_shape_and_cnt(conv_name_list,model_list_dict,channel_info_list):
    
    chanel={}      
    shape={}
    cnt_o={}
    cnt_i={}
    for name in conv_name_list:
        all_channel_num=0
        if name[0:len(name)-7] in channel_info_list[0]:            
            rm_channel=channel_info_list[0][name[0:len(name)-7]]            
            all_channel_num=len(rm_channel)+model_list_dict[0][name].shape[0]   
            shape[name]=[all_channel_num,0]  
            # shape[name[0:len(name)-6]+'bias']= [all_channel_num]  
            chanel[name]=list(range(shape[name][0]))            
            cnt_o[name]=[len(channel_info_list)]*all_channel_num
        else:
            rm_channel=[]
            s=[0,0]
            all_channel_num=s[0]=model_list_dict[0][name].shape[0]
            shape[name]=s
            # shape[name[0:len(name)-6]+'bias']= s[0]              
            chanel[name]=list(range(s[0]))            
            cnt_o[name]=[len(channel_info_list)]*all_channel_num
            continue
        for i in range(1,len(channel_info_list)):
            if name[0:len(name)-7] in channel_info_list[i]:
                rm_channel=[val for val in rm_channel if val in  channel_info_list[i][name[0:len(name)-7]]]          
            else:
                rm_channel=[]
                s=[0,0]
                s[0]=model_list_dict[i][name].shape[0]
                shape[name]=s
                # shape[name[0:len(name)-6]+'bias']= s[0]                  
                chanel[name]=list(range(shape[0]))
                break
        # print(name)
        # print(chanel[name])
        # print(rm_channel)
        # print(model_list_dict)
        # for m in model_list_dict:
        #     print(m[name].shape)
        # for m in channel_info_list:
        #     print(len(m[name[0:len(name)-7]]))
        for x in rm_channel:
            chanel[name].remove(x)
        shape[name][0]=len(chanel[name])
        # print(cnt_o[name])
        rm_cnt=[0]*all_channel_num
        for ch in channel_info_list:
            if name[0:len(name)-7] in ch:
                for i in range(len(cnt_o[name])):
                    if i in ch[name[0:len(name)-7]]:
                        rm_cnt[i]+=1
        for i in range(len(rm_cnt)):
            cnt_o[name][i]=cnt_o[name][i]-rm_cnt[i]



    cnt_last=[2]*3
    out_last=3
    for name in conv_name_list:
        shape[name][1]=out_last
        shape[name[0:len(name)-6]+'bias']= [shape[name][0]]
        cnt_i[name]=cnt_last
        cnt_last=cnt_o[name]
        out_last=shape[name][0]


    # print(chanel)
    # for name in chanel:
    #     print(name)     
    #     print(shape[name])   
    #     # print(cnt_o[name])
    #     # print(cnt_i[name])
    #     print(chanel[name])
    
    # # print(shape)
    # # print("h")
    return chanel,shape,cnt_o,cnt_i





        # shape=[0,0]
        
        # shape[0]=model_list_dict[0][name].shape[0]        
        # # print(channel_info_list[0].keys())
        # for i in range(1,len(channel_info_list)):

        #     if name[0:len(name)-7] in channel_info_list[0]:
            
            
        #     rm_channel=channel_info_list[0][name[0:len(name)-7]]
        #     for x in rm_channel:
        #         chanel.remove(x)            
        # else:
        #     chanel=list(range(shape[0]))

        # print(shape)
        # print(chanel)
        # for i in range(1,len(model_list_dict)):


def find_index(x,num):
    if x in num:
        return -1    
    for i in range(len(num)):
        if num[i]>x:
            return i
    return len(num)
def pruned_model_ave(model_list,channel_info_list):

    # dst_model=.copy()
    print(channel_info_list)
    dst_model=copy.deepcopy(model_list[0])
    dst_model=model_para_clear(dst_model)
    model_list_dict=[]
    dst_model_dict=dict(dst_model.named_parameters())
    for i in range(len(model_list)):
        params=model_list[i].named_parameters()
        model_list_dict.append(dict(params))
    # l =list(model_list_dict[0].keys())
    conv_cnt={}
    conv_name_list=get_model_conv_name(dst_model)
    print(conv_name_list)
    channel_conv,shape_conv, _ ,_ =get_aved_conv_shape_and_cnt(conv_name_list,model_list_dict,channel_info_list)
    print(channel_conv)
    print(shape_conv)

    for name in dst_model_dict:
        print(name)
        print(dst_model_dict[name].shape)
        if len(dst_model_dict[name].shape)>1 and 'features' in name:
            dst_model_dict[name]=torch.zeros((shape_conv[name][0],shape_conv[name][1],3,3))
            print(dst_model_dict[name].shape)
            for i in range(len(model_list_dict)):
                model=model_list_dict[i]
                model_w = model[name]
                # print(model_w.shape)
                # print()

                i_name=conv_name_list[conv_name_list.index(name)-1]
                if name[0:len(name)-7] not in channel_info_list[i] and i_name[0:len(i_name)-7] not in channel_info_list[i]:
                    dst_model_dict[name]+=model_w
                    break
                for o_c in range(dst_model_dict[name].shape[0]):

                    if name[0:len(name)-7] in channel_info_list[i]: #该层有被剪枝
                        del_index=find_index(o_c,channel_info_list[i][name[0:len(name)-7]])
                        if del_index!=-1:
                            o_index=o_c-del_index
                        else: #这个通道被减掉了，不需要加了
                            continue
                    else:
                        o_index=o_c        
                    # print(o_c)
                    # print(channel_info_list[i][name[0:len(name)-7]])                                         
                    # print(o_index)
                    if i_name[0:len(i_name)-7] in channel_info_list[i]: #输入通道有被剪枝
                        for i_c in range(dst_model_dict[name].shape[1]):
                            del_index=find_index(i_c,channel_info_list[i][i_name[0:len(i_name)-7]])
                            if del_index!=-1:
                                o_index=o_c-del_index
                            else: #这个通道被减掉了，不需要加了
                                continue

                    else: #输出被剪枝，输入没有
                        dst_model_dict[name][o_c]+=model_w[o_index]
        elif 'features' in name and len(dst_model_dict[name].shape)==1:
            print(name)

            tmp=name.split('.')
            tmp_name=tmp[0]+'.'+tmp[1]+'.weight'
            while(True):
                print(tmp_name)
                if tmp_name  in shape_conv:
                    dst_model_dict[name]=torch.zeros((shape_conv[tmp_name][0]))
                    print(dst_model_dict[name].shape)
                    print("-------------------")
                    break
                else:
                    tmp[1]=int(tmp[1])
                    tmp[1]-=1
                    tmp_name=tmp[0]+'.'+str(tmp[1])+'.weight'
            
            for i in range(len(model_list_dict)):
                model=model_list_dict[i]
                model_w = model[name]
                print(model_w.shape)
                
                if tmp_name[0:len(tmp_name)-7] not in channel_info_list[i]:
                    dst_model_dict[name]+=model_w
                else:
                    for j in range(dst_model_dict[name].shape[0]):  
                        del_index=find_index(j,channel_info_list[i][tmp_name[0:len(tmp_name)-7]]) 
                        if del_index!=-1:
                            o_index=j-del_index
                        else: #这个通道被减掉了，不需要加了
                            continue
                        dst_model_dict[name][j]+=model_w[o_index]  
        elif 'classifier.0.weight' in name:
            print(name)

            tmp=name.split('.')
            tmp_name='features.40.weight'  
              
            dst_model_dict[name]=torch.zeros((model_list_dict[0][name].shape[0],shape_conv[name][0]))
            print(dst_model_dict[name].shape)  
            for i in range(len(model_list_dict)):
                model=model_list_dict[i]
                model_w = model[name]
                if tmp_name[0:len(tmp_name)-7] not in channel_info_list[i]:
                    dst_model_dict[name]+=model_w
                else:
                    for j in range(dst_model_dict[name].shape[0]):    

                        dst_model_dict[name][j]+=model_w[channel_conv[tmp_name][j]]              


            #     if i_name in channel_info_list[i]:
            #         dst_model_dict[name]+=model_w

            # for o_c in range(dst_model_dict[name].shape[0]):
            #     for i_c in range(dst_model_dict[name].shape[1]):



            # for i in range(len(model_list_dict)):
            #     model=model_list_dict[i]
            #     model_w = model[name]            
            #     print(model_w.shape)
            #     if len(model_w.shape)>1:
            #         for o_c in range(model_w.shape[0]):
            #             for i_c in range(model_w.shape[1]):
            #                 print(model_w[0][0].shape)
    # print(channel_info_list[0].keys())
    # print((model_name))

    # params1=model1.named_parameters()
    # dict_dst_params1 = dict(params1)  
    # params2=model2.named_parameters()
    # dict_dst_params2 = dict(params2)  
    # # print(type(params2))
    # for name in dict_dst_params1:       
    #     print(name)
    #     if name in dict_dst_params2:
    #         print("origin",dict_dst_params1[name].shape)
    #         print("pruned",dict_dst_params2[name].shape)
    #         # print("pruned",dict_dst_params2[name])
    #     else:
    #         print("origin",dict_dst_params1[name].shape)
    #         print("pruned","all delete")
def get_channel_info(filename):
    channel_info={}
    with open(filename, 'r') as f:
        lines = f.readlines()
        layer=[]
        for line in lines:
            tmp=line.split(':')
            # print(tmp[0])
            # layer.append(tmp[0])
            l=tmp[1][1:len(tmp[1])-2].split(',')
            l = list(map(int, l))
            l.sort()
            # layer.append(l)
            # channel_info.append(layer)
            channel_info[tmp[0]]=l
    return channel_info


def model_complete_size(model,channel_info):
    # print(channel_info)
    dst_model_dict=dict(model.named_parameters())
    model_layers_size={} 
    model_layers_remaind_channel={}
    model_layers_deleted_channel={}

    name_list=[]
    name_cnt=-1
    for name in dst_model_dict:       
        # print(name)
        # print(dst_model_dict[name].shape)
        
        model_layers_size[name]=list(dst_model_dict[name].shape)
        name_list.append(name)
        name_cnt+=1
        tmp=name.split('.')
        name_tmp=tmp[0]+'.'+tmp[1]
        if 'features' in name and 'weight' in name: #卷积权重层
            if len(dst_model_dict[name].shape)>1:
                if name_tmp in channel_info:
                    model_layers_size[name][0]+=len(channel_info[name_tmp])
                    if name_cnt>1:
                        model_layers_size[name][1]=model_layers_size[name_list[name_cnt-2]][0]

                    model_layers_deleted_channel[name]=channel_info[name_tmp]
                    model_layers_remaind_channel[name]=list(range(0,model_layers_size[name][0]))
                    for x in model_layers_deleted_channel[name]:
                        model_layers_remaind_channel[name].remove(x)
                if name_tmp not in channel_info:
                    model_layers_deleted_channel[name]=-1
                    model_layers_remaind_channel[name]=list(range(0,model_layers_size[name][0]))
            else:
                while True:
                    tmp[1]=int(tmp[1])
                    tmp[1]-=1
                    name_tmp = tmp[0]+'.'+str(tmp[1])+'.bias'
                    if name_tmp in model_layers_size:
                        model_layers_size[name]=model_layers_size[name_tmp]
                        model_layers_deleted_channel[name]=model_layers_deleted_channel[name_tmp]
                        model_layers_remaind_channel[name]=model_layers_remaind_channel[name_tmp] 
                        break

        elif 'features' in name and 'bias' in name: 
            name_tmp = name_tmp+'.weight'
            model_layers_size[name][0]=model_layers_size[name_tmp][0]
            model_layers_deleted_channel[name]=model_layers_deleted_channel[name_tmp]
            model_layers_remaind_channel[name]=model_layers_remaind_channel[name_tmp]  
        else: 
            if len(dst_model_dict[name].shape)>1:
                model_layers_size[name][1]=model_layers_size[name_list[name_cnt-2]][0]
            model_layers_deleted_channel[name]=-1
            model_layers_remaind_channel[name]=list(range(0,model_layers_size[name][0]))
    return model_layers_size,model_layers_deleted_channel,model_layers_remaind_channel,name_list



def model_zeros_fill(model_in,model,channel_info):

    model_layers_size,model_layers_deleted_channel,model_layers_remaind_channel,name_list=model_complete_size(model,channel_info)
    resized_model=copy.deepcopy(model_in)
    # print(model_layers_size)
    # resized_model=model_in.copy()
    # model_para_print(resized_model)
    dst_model_dict=dict(resized_model.named_parameters())
    src_model_dict=dict(model.named_parameters())
    name_cnt=-1
    for name in dst_model_dict:       
        name_cnt+=1
        # print(name)
        with torch.no_grad():
            if len(src_model_dict[name].shape)>1:
                if src_model_dict[name].shape[1]==model_layers_size[name][1]: # 输入一样
                    if src_model_dict[name].shape[0]==model_layers_size[name][0]: #输出一样
                        dst_model_dict[name].set_(src_model_dict[name])
                        # print(dst_model_dict[name])
                    else:#输出不一样
                        set_tmp=torch.zeros(model_layers_size[name])
                        
                        # print(dst_model_dict[name].shape)
                        for out_index in range(len(model_layers_remaind_channel[name])):
                            out_channel=model_layers_remaind_channel[name][out_index]
                            set_tmp[out_channel]=src_model_dict[name][out_index]
                            
                        dst_model_dict[name].set_(set_tmp)
                        # print(dst_model_dict[name])
                else:
                    
                    set_tmp=torch.zeros(model_layers_size[name])

                    last_name=name_list[name_cnt-2]
                    len_out=len(model_layers_remaind_channel[name])
                    len_in = len(model_layers_remaind_channel[last_name])
                    for out_index in range(len_out):
                        out_channel=model_layers_remaind_channel[name][out_index]
                        for in_index in range(len_in):                            
                            in_channel=model_layers_remaind_channel[last_name][in_index]                            
                            set_tmp[out_channel][in_channel]=src_model_dict[name][out_index][in_index]
                    dst_model_dict[name].set_(set_tmp)
                    # print(dst_model_dict[name])
            else:
                if src_model_dict[name].shape[0]==model_layers_size[name][0]: #输出一样
                    dst_model_dict[name].set_(src_model_dict[name])
                    # print(dst_model_dict[name])
                else:#输出不一样
                    len_out=len(model_layers_remaind_channel[name])                    
                    set_tmp=torch.zeros(model_layers_size[name])

                    # print(dst_model_dict[name])
                    for out_index in range(len_out):
                        # print(src_model_dict[name][out_index])
                        # print(model_layers_remaind_channel[name][out_index])
                        set_tmp[model_layers_remaind_channel[name][out_index]]=src_model_dict[name][out_index]
                    # print(set_tmp)
                    dst_model_dict[name].set_(set_tmp)  
                    # print(dst_model_dict[name])
                # print(dst_model_dict[name])
            # print(model_layers_size[name])
            # print(model_layers_deleted_channel[name])
            # print(model_layers_remaind_channel[name])
    # print('-----------------------------------------------------------')
    # model_para_print(resized_model)
    return resized_model
    # print('1')

def model_list_avg(dst_models,src_model_list):

    '''
    b1=mbnn_fl.model_para_set(mbnn_fl.B_LetNet(),1)
    b2=mbnn_fl.model_para_set(mbnn_fl.B_LetNet_b1(),2)
    b3=mbnn_fl.model_para_set(mbnn_fl.B_LetNet_b2(),4)
    b4=mbnn_fl.model_para_set(mbnn_fl.B_LetNet_b3(),6)
    # mbnn_fl.model_para_print(b1)
    c=mbnn_fl.B_LetNet()
    c=mbnn_fl.model_list_avg(c,[b1,b2,b3,b4])
    mbnn_fl.model_para_print(c)
    '''
    # dst_models=dst_models.to('cpu')
    dst_models=model_para_clear(dst_models)    
    dst_params=dst_models.named_parameters()
    dict_dst_params = dict(dst_params)

    for name in dict_dst_params:        
        cnt=0
        with torch.no_grad():
            for src_model in src_model_list:
                src_params=src_model.named_parameters()
                dict_src_params = dict(src_params)
                if name in dict_src_params:
                    cnt+=1.0
                    dict_dst_params[name].set_(dict_src_params[name]+dict_dst_params[name])
            if cnt!=0:
                dict_dst_params[name].set_(dict_dst_params[name]/cnt)
    return dst_models
def model_cp(dst_model,src_model):

    src_params=src_model.named_parameters()
    dst_params=dst_model.named_parameters()

    dict_dst_params = dict(dst_params)
    dict_src_params = dict(src_params)
    for name in dict_dst_params:
        with torch.no_grad():
            # print(name)
            # print(dict_dst_params[name])
            # print(p)
            
            dict_dst_params[name].set_(dict_src_params[name])

    return dst_model


def prune_network(network=None):
    args = get_parameter()    
    layer_idx = 13
    args.prune_layers = ['conv0', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'conv9', 'conv10', 'conv11',
                         'conv12', 'conv13']
    args.network = 'vgg16_bn'
    args.data_set = 'CIFAR10'
    args.data_path = './'
    args.save_path = './'
    args.load_path = './trained_models/check_point_CIFAR100.pth'
    args.update_param_epoch = 1
    args.finetune_epoch = 1
    args.ratio = 0.35
    args.u = 1
    args.flops_ratio = 0.3
    args.batch_size = 256
    args.gpu_no=-1
    
    model = VGG(args.network, args.data_set)
    old_model=copy.deepcopy(model)
    network, prune_index = prune_vgg.prune_network(args, network=model)
    
    final_index = FinalIndex(old_model, prune_index, layer_idx)
    mask_all = final_index.init_mask(args)
    mask_final = final_index.get_all_conv(mask_all)
    final_pruned_index = final_index.get_final_pruned_index(mask_final, mask_all)

    print(final_pruned_index)
    return network,final_pruned_index