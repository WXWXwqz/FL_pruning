import argparse
import  numpy as np
import torch
from syft.workers.websocket_server import WebsocketServerWorker
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import syft as sy
import sys
import socket
import re
import socket
import resource
from utils import  get_data_set

def limit_memory(maxsize):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

def get_host_ip():
  try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    ip = s.getsockname()[0]
  finally:
    s.close()
 
  return ip
# def main(**kwargs):  
#     datasetname=kwargs['datasetname']
#     data=np.load('data/'+datasetname)['data']
#     target=np.load('data/'+datasetname)['label']
#     # Create websocket worker
#     worker = WebsocketServerWorker(**kwargs)   
#     data=data.view(data.shape[0],1,28,28)/255.0
#     dataset = sy.BaseDataset(data, target)
#     worker.add_dataset(dataset, key="mnist_iid") 
#     worker.start()

#     return worker


if __name__ == "__main__":
    
    hook = sy.TorchHook(torch)
    # Arguments
    parser = argparse.ArgumentParser(description="Run websocket server worker.")
    parser.add_argument(
        "--port", "-p", type=int, help="port number of the websocket server worker, e.g. --port 8777"
    )
    parser.add_argument("--host", type=str, default="localhost", help="host for the connection")
    parser.add_argument("--dataset", type=str, default="mnist_iid", help="dataset for need train")

    parser.add_argument(
        "--id", type=str, help="name (id) of the websocket server worker, e.g. --id bob"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket server worker will be started in verbose mode",
    )    
    print('hello \n federated learning')
    args = parser.parse_args()
    print('worker name:',args.id)
    host = get_host_ip()
    # limit_memory(1024*1024*1024*2)
    
    num =re.sub("\D", "", args.id)
    datasetname=args.dataset+'_'+str(num)+'.npz'
    
    if(args.dataset=='cifar10_iid'):
        ports=str(7000+int(num))


    print('work host:'+host+':'+ports)
    kwargs = {
        "id": args.id,
        "host": host,
        "port": ports,
        "hook": hook,
        "verbose": args.verbose,
    }
    print('worker dataset:'+datasetname)

    data=np.load('data/'+datasetname)['data']
    target=np.load('data/'+datasetname)['label']
    print(type(data))
    print(type(target))
    
    # Create websocket worker
    worker = WebsocketServerWorker(**kwargs)   
    data=torch.tensor(data,dtype=torch.float32)
    target=torch.tensor(target,dtype=torch.long)
    if(args.dataset=='mnist_iid' or args.dataset=='mnist_non_iid'):
        data=data.view(data.shape[0],1,28,28)/255.0
    if 'cifar10' in args.dataset:
        data_set = get_data_set(train_flag=False)
        # data=data.view(data.shape[0],1,28,28)/255.0
        dataset=data_set
    # print(dataset.shape)
    # print(data[0])
    # print(target.shape)
    # print(target)
    # dataset = sy.BaseDataset(data, target)
    print('datasetname',args.dataset)
    worker.add_dataset(dataset, key=args.dataset) 
    worker.start()
    # main(**kwargs)
