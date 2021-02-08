'''
federated learning implement by muilt-proceessing
test accuracy of CNN in FL by IID-MNIST
'''
import multiprocessing
import time
from multiprocessing import Process, Queue
import threading
import time
import numpy as np
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
import syft as sy
from syft import workers
import log
import socket
import sys
import re
import random
from network import VGG
import fl_func

@torch.jit.script
def loss_fn(pred, target):
    return F.cross_entropy(input=pred, target=target)
# torch.jit.RecursiveScriptModule

if __name__ == "__main__":

    
    device = torch.device("cuda")
    torch.manual_seed(1)
    
    epochs = 1
    LOG_FILE ='fl_ctl_syn_vgg_iid_cifar10'+str(epochs)+socket.gethostname()+str(device)+time.strftime('%Y-%m-%d',time.localtime(time.time()))+'.txt'
    log=log.Log(LOG_FILE,True)
    log.log(LOG_FILE)
    log.log('Federated learning')
    workers_num=1
    fl_func.prune_network()
    hook = sy.TorchHook(torch)  # hook torch as always :)
    
    mock_data = torch.zeros(100,3,32,32)
    mock_data.to(device)
    ave_once_num=1
    log.log()
    optimizer = "SGD"
    batch_size = 100
    optimizer_args = {"lr" : 0.01, "weight_decay" : 0.001}
    
    shuffle = True
    
    log.log(optimizer,optimizer,optimizer_args,epochs)
    kwargs_websocket = {"host": "192.168.123.166", "hook": hook, "verbose": False}  
    model=VGG().to(device)
    model_ave=VGG().to(device)
    fl_func.prune_network()
    # model
    worker_client=[0]*workers_num
    train_config=[0]*workers_num
    for x in range(10):
        log.log("train num,",x)    
        model=VGG().to(device)
        model_ave=VGG().to('cpu')
        end_flg=0        
        model_list=[0]*workers_num
        traced_model=[0]*workers_num
        # train_config=[0]*workers_num
        worker_loss=[0]*workers_num
        ports=[0]*workers_num
        for i in range(workers_num):
            model_list[i]=model.copy()
            ports[i]=7000+i
            traced_model[i]=torch.jit.trace(model_list[i], mock_data.to(device))
            
            if x==0:
                train_config[i] = sy.TrainConfig(model=traced_model[i],
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                batch_size=batch_size,
                                optimizer_args=optimizer_args,
                                epochs=epochs,
                                shuffle=shuffle)
                worker_client[i] = workers.websocket_client.WebsocketClientWorker(id='worker'+str(i), port=ports[i], **kwargs_websocket)
                train_config[i].send(worker_client[i]) 
            train_config[i].model=traced_model[i]
            
        merge_cnt=0
        
        cnt=0
        while(True):
            random_worker_list=random.sample(range(0,workers_num),ave_once_num)
            # random_worker_list=[3]
            log.log('worker in this ave:',random_worker_list)
            for i in random_worker_list:
                # train_config[i].model_ptr= train_config[i].model.send(worker_client[i])   
                train_config[i].send_model(worker_client[i])    
                log.log('worker ',i)      
                # for kkktt in range(5):                                      
                loss=worker_client[i].fit(dataset_key="cifar10_iid",device=device) 
                worker_loss[i]=loss
                # while loss >0.1:
                #     loss=worker_client[i].fit(dataset_key="mnist_non_iid")   
                log.log(loss)
                # while loss>0.5:
                #     loss=worker_client[i].fit(dataset_key="mnist_non_iid")   
                #     print(loss)
            loss_min= sum(worker_loss)/workers_num
            log.log("loss_min {:.4f}".format(loss_min))
            for traincnt in range(workers_num):
                # if worker_loss[traincnt]>=loss_min*1.5 and worker_loss[traincnt]>0.05:
                #     log.log("again",traincnt)
                #     loss_again=worker_client[traincnt].fit(dataset_key="beardata_non_iid",device=device)  
                #     log.log("loss_again:{:.4f}".format(loss_again))
                #     while loss_again >loss_min*1.5 and worker_loss[traincnt]>0.05:                
                #         loss_again=worker_client[traincnt].fit(dataset_key="beardata_non_iid",device=device)   
                #         worker_loss[traincnt]=loss_again
                #         log.log("loss_again:{:.4f}".format(loss_again))
                getmodel=train_config[traincnt].model_ptr.get().obj
                print("model get")
                # mbnn_fl.model_para_print(getmodel)
                # model=VGG().to(device)
                model=VGG().to(device)
                model=fl_func.model_cp(model,getmodel)   
                model,channel_index= fl_func.prune_network(model)


                model_list[traincnt]=model.to('cpu')

            model_ave=fl_func.model_list_avg(model_ave,model_list)
            model_ave=model_ave.to(device)
            train_config[i].model=torch.jit.trace(model_ave, mock_data.to(device))

            cnt+=1
            log.log('merger, ',cnt)
      








        # if(sum(train_complete_flag)==workers_num):
        #     model_ave=mbnn_fl.LetNet5()
        #     model_ave=mbnn_fl.model_list_avg(model_ave,model_list)  
        #     # model_ave=utils.federated_avg(model_ave)
        #     merge_cnt+=1
        #     log.log('meger num:',merge_cnt)
        #     torch.save(model_ave.state_dict(), 'non_iid_mnist'+time.strftime('%Y-%m-%d',time.localtime(time.time()))+"mnist_mbnn_iid_FL.pt")            
        #     for i in range(workers_num):
        #         train_complete_flag[i]=0
        #         q_main[i].put('train')
        #         q_main[i].put(model_ave)         
        #     # print('*****************************')  
        #     mbnn_fl.test_model(model=model_ave,loss_fn=loss_fn,log=log,device=device)  
          
        # else:
        #     for i in range(workers_num):
        #         if not q_sub[i].empty():
        #             tmp=q_sub[i].get()
        #             print(tmp)
        #             if tmp=='model':
        #                 model_list[i]=q_sub[i].get()
        #                 train_complete_flag[i]=1
            
    # print ("p.pid:", p.pid)
    # print ("p.name:", p.name)
    # print ("p.is_alive:", p.is_alive())

    # while(p.is_alive()):
    #     pass