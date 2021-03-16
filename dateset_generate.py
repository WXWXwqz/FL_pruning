#generate  MNIST IID data for 12 worker
import  numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import scipy.io as scio
from collections import Counter
def generate_beardata_iid(worker_num):

    data_origin=np.load('bear_train_dataset.npz')['data']
    targets=np.load('bear_train_dataset.npz')['label']

    length=data_origin.shape[0]/worker_num
    for i in range(worker_num):    
        data=data_origin[int(i*length):int(length*(i+1))]
        label=targets[int(i*length):int(length*(i+1))]
        np.savez('beardata_iid_'+str(i)+'.npz',data=data,label=label)
def generate_beardata_non_iid(worker_num):
    data_origin=np.load('bear_train_dataset.npz')['data']
    targets_origin=np.load('bear_train_dataset.npz')['label']

    length=data_origin.shape[0]/worker_num
    data=np.zeros(data_origin.shape)
    label=np.zeros(targets_origin.shape)
    index=0
    print(data_origin.shape)
    for i in range(10):        
        indices = np.isin(torch.tensor(targets_origin), i).astype("uint8")    
        # selected_data = torch.masked_select(torch.tensor(data_origin), torch.tensor(indices))   
        selected_data = (
            torch.masked_select(torch.tensor(data_origin).transpose(0,1), torch.tensor(indices))
            .view(2000, -1)
            .transpose(1, 0)
        )  
        selected_targets = torch.masked_select(torch.tensor(targets_origin), torch.tensor(indices))
        data[index:index+selected_targets.shape[0]]=selected_data
        label[index:index+selected_targets.shape[0]]=selected_targets
        index+=selected_targets.shape[0]

    length=data.shape[0]/worker_num
    for i in range(worker_num):    
        data_save=data[int(i*length):int(length*(i+1))]
        label_save=label[int(i*length):int(length*(i+1))]
        np.savez('beardata_non_iid_'+str(i)+'.npz',data=data_save,label=label_save)


def generate_bear_train_test_dateset():
    path = 'beardateset.mat'
    beardata = scio.loadmat(path)
    data=beardata['traindata'] 
    label = beardata['traindatalab']
    index = [i for i in range(len(data))] 
    np.random.shuffle(index)
    data = data[index]
    label = label[index]
    label=label.reshape(label.shape[0])
    np.savez('bear_train_dataset.npz',data=data[0:40000],label=label[0:40000])
    np.savez('bear_test_dataset.npz',data=data[40000:470000],label=label[40000:47000])

def generate_bear_train_verif_test_dateset():
    path = 'beardateset.mat'
    beardata = scio.loadmat(path)
    data=beardata['traindata'] 
    label = beardata['traindatalab']
    index = [i for i in range(len(data))] 
    np.random.shuffle(index)
    data = data[index]
    label = label[index]
    label=label.reshape(label.shape[0])
    np.savez('bear_train_dataset.npz',data=data[0:int(len(data)*0.7)],label=label[0:int(len(data)*0.7)])
    np.savez('bear_verif_dataset.npz',data=data[int(len(data)*0.7):int(len(data)*0.8)],label=label[int(len(data)*0.7):int(len(data)*0.8)])
    np.savez('bear_test_dataset.npz',data=data[int(len(data)*0.8):470000],label=label[int(len(data)*0.8):47000])

def generate_mnist_non_iid_singe_pro(worker_num):
    mnist_dataset = torchvision.datasets.MNIST(root='data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
    public_pro=0.005
    public_data=mnist_dataset.data[0:int(mnist_dataset.data.shape[0]*public_pro)]
    public_label=mnist_dataset.targets[0:int(mnist_dataset.data.shape[0]*public_pro)]
    client_data=mnist_dataset.data[int(mnist_dataset.data.shape[0]*public_pro):mnist_dataset.data.shape[0]]
    client_label=mnist_dataset.targets[int(mnist_dataset.data.shape[0]*public_pro):mnist_dataset.data.shape[0]]

    data=np.zeros(client_data.shape)
    label=np.zeros(client_label.shape)

    index=0
    for i in range(10):        
        indices = np.isin(client_label, i).astype("uint8")    
        selected_data = (
            torch.masked_select(client_data.transpose(0,2), torch.tensor(indices))
            .view(28, 28, -1)
            .transpose(2, 0)
        )   
        selected_targets = torch.masked_select(client_label, torch.tensor(indices))

        data_save=np.concatenate((selected_data,public_data),axis=0)
        
        print('Counter(data)\n',Counter(np.array(selected_targets))) 
        label_save=np.concatenate((selected_targets,public_label),axis=0)
        # print(data_save)
        # print(label_save)
        np.savez('data/'+'mnist_non_iid_'+str(i)+'.npz',data=data_save,label=label_save)

def generate_bear_non_iid_singe_pro(worker_num):
    
    data_origin=np.load('bear_train_dataset.npz')['data']
    targets_origin=np.load('bear_train_dataset.npz')['label']
    print(data_origin.shape)
    
    # print(data_origin.shape)
    public_pro=0.03
    public_data=data_origin[0:int(data_origin.shape[0]*public_pro)].reshape(int(data_origin.shape[0]*public_pro),1,2000)
    public_label=targets_origin[0:int(targets_origin.shape[0]*public_pro)]
    
    client_data=data_origin[int(data_origin.shape[0]*public_pro):data_origin.shape[0]]
    client_label=public_label[int(public_label.shape[0]*public_pro):public_label.shape[0]]

    data=np.zeros(client_data.shape)
    label=np.zeros(client_label.shape)

    index=0
    for i in range(worker_num):        
        indices = np.isin(torch.tensor(targets_origin), i).astype("uint8")    
        # selected_data = torch.masked_select(torch.tensor(data_origin), torch.tensor(indices))   
        selected_data = (
            torch.masked_select(torch.tensor(data_origin).transpose(0,1), torch.tensor(indices))
            .view(2000, -1)
            .transpose(1, 0)
        )  
        selected_targets = torch.masked_select(torch.tensor(targets_origin), torch.tensor(indices))
        
        
        data_save=np.concatenate((selected_data.reshape(selected_data.shape[0],1,2000),public_data),axis=0)
        print(data_save.shape)
        print('Counter(data)\n',Counter(np.array(selected_targets))) 
        label_save=np.concatenate((selected_targets,public_label),axis=0)
        print('Counter(data)\n',Counter(np.array(label_save))) 
        print('number\n',label_save.shape) 
        # print(data_save)
        # print(label_save)
        #np.savez('data/'+'mnist_non_iid_'+str(i)+'.npz',data=data_save,label=label_save)    
        np.savez('beardata_non_iid_'+str(i)+'.npz',data=data_save,label=label_save)      
def generate_bear_non_iid_mbnn_singe_pro(worker_num):
    
    branch1_label=[0,3,7,9]
    branch2_label=[1,6,8]
    branch3_label=[2,4,5]
    branch_label=[branch1_label,branch2_label,branch3_label]
 

    branch_label=[branch1_label,branch2_label,branch3_label]
    data_origin=np.load('bear_train_dataset.npz')['data']
    targets_origin=np.load('bear_train_dataset.npz')['label']
    print(data_origin.shape)
    
    # print(data_origin.shape)
    public_pro=0.03
    public_data=data_origin[0:int(data_origin.shape[0]*public_pro)].reshape(int(data_origin.shape[0]*public_pro),1,2000)
    public_label=targets_origin[0:int(targets_origin.shape[0]*public_pro)]
    
    client_data=data_origin[int(data_origin.shape[0]*public_pro):data_origin.shape[0]]
    client_label=public_label[int(public_label.shape[0]*public_pro):public_label.shape[0]]

    data=np.zeros(client_data.shape)
    label=np.zeros(client_label.shape)

    index=0
    for i in range(worker_num):        
        indices = np.isin(torch.tensor(targets_origin), i).astype("uint8")    
        # selected_data = torch.masked_select(torch.tensor(data_origin), torch.tensor(indices))   
        selected_data = (
            torch.masked_select(torch.tensor(data_origin).transpose(0,1), torch.tensor(indices))
            .view(2000, -1)
            .transpose(1, 0)
        )  
        selected_targets = torch.masked_select(torch.tensor(targets_origin), torch.tensor(indices))
        
        
        data_save=np.concatenate((selected_data.reshape(selected_data.shape[0],1,2000),public_data),axis=0)
        print(data_save.shape)
        print('Counter(data)\n',Counter(np.array(selected_targets))) 
        label_save=np.concatenate((selected_targets,public_label),axis=0)
        print('Counter(data)\n',Counter(np.array(label_save))) 
        b=0
        for k in range(3):
            if i in branch_label[k]:
                print("work {} in branch {}".format(i,k))
                b=k


        for j in range(label_save.shape[0]):
            if label_save[j]  in branch_label[b]:
                # print(label_save[j])
                t=branch_label[b].index(label_save[j])
                label_save[j]=t                
                # print(label_save[j])
                
            else:
                label_save[j]=len(branch_label[b])
        print('Counter(data)\n',Counter(np.array(label_save))) 
        print('number\n',label_save.shape) 
        # print(data_save)
        # print(label_save)
        #np.savez('data/'+'mnist_non_iid_'+str(i)+'.npz',data=data_save,label=label_save)    
        np.savez('beardata_non_mbnn_iid_'+str(i)+'.npz',data=data_save,label=label_save)      
def generate_mnist_non_iid(worker_num):
    mnist_dataset = torchvision.datasets.MNIST(root='data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
    public_pro=0
    public_data=mnist_dataset.data[0:int(mnist_dataset.data.shape[0]*public_pro)]
    public_label=mnist_dataset.targets[0:int(mnist_dataset.data.shape[0]*public_pro)]
    client_data=mnist_dataset.data[int(mnist_dataset.data.shape[0]*public_pro):mnist_dataset.data.shape[0]]
    client_label=mnist_dataset.targets[int(mnist_dataset.data.shape[0]*public_pro):mnist_dataset.data.shape[0]]

    data=np.zeros(client_data.shape)
    label=np.zeros(client_label.shape)

    index=0
    for i in range(10):        
        indices = np.isin(client_label, i).astype("uint8")    
        selected_data = (
            torch.masked_select(client_data.transpose(0,2), torch.tensor(indices))
            .view(28, 28, -1)
            .transpose(2, 0)
        )   
        selected_targets = torch.masked_select(client_label, torch.tensor(indices))
        data[index:index+selected_targets.shape[0]]=selected_data
        label[index:index+selected_targets.shape[0]]=selected_targets
        index+=selected_targets.shape[0]

    length=data.shape[0]/worker_num
    for i in range(worker_num):    
        data_save=data[int(i*length):int(length*(i+1))]
        # print(data_save)
        # print(public_data)
        # data_save=torch.Tensor(data_save,dtype=torch.float32)
        # public_data=torch.Tensor(public_data,dtype=torch.float32)
       
        data_save=np.concatenate((data_save,public_data),axis=0)
        label_save=label[int(i*length):int(length*(i+1))]
        print('Counter(data)\n',Counter(label_save)) 
        label_save=np.concatenate((label_save,public_label),axis=0)
        # print(data_save)
        # print(label_save)
        np.savez('mnist_non_iid_'+str(i)+'.npz',data=data_save,label=label_save)
def generate_mnist_non_iid_singe(worker_num):
    mnist_dataset = torchvision.datasets.MNIST(root='data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
   
   
    for i in range(10):       
        indices = np.isin(mnist_dataset.targets, i).astype("uint8")   
        selected_data = (
            torch.masked_select(mnist_dataset.data.transpose(0,2), torch.tensor(indices))
            .view(28, 28, -1)
            .transpose(2, 0)
        )   
        selected_targets = torch.masked_select(mnist_dataset.targets, torch.tensor(indices))
        np.savez('data/'+'mnist_non_iid_'+str(i)+'.npz',data=selected_data,label=selected_targets)

        print('Counter(data)\n',Counter(np.array(selected_targets))) 

def generate_mnist_iid(worker_num):

    mnist_dataset = torchvision.datasets.MNIST(root='data/',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)

    length=mnist_dataset.data.shape[0]/worker_num
    for i in range(worker_num):    
        data=mnist_dataset.data[int(i*length):int(length*(i+1))].numpy()
        label=mnist_dataset.targets[int(i*length):int(length*(i+1))].numpy()
        np.savez('mnist_iid_'+str(i)+'.npz',data=data,label=label)

def generate_cifar10_iid(worker_num):

    cifar10_dataset = torchvision.datasets.CIFAR10(root='data/',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)

    length=cifar10_dataset.data.shape[0]/worker_num
    for i in range(worker_num):    
        data=cifar10_dataset.data[int(i*length):int(length*(i+1))]
        label=cifar10_dataset.targets[int(i*length):int(length*(i+1))]
        np.savez('data/cifar10_iid_'+str(i)+'.npz',data=data,label=label)


def loader(dataset_name):
    d=np.load(dataset_name)['data']
    l=np.load(dataset_name)['label']
    print(d.shape)
    print(l)
    cnt=0
    for i in range(len(l)):
        if l[i]==1:
            cnt+=1
        else:
            print(cnt)
            break


    dataset=TensorDataset(torch.tensor(d),torch.tensor(l))
    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=100,
                                            shuffle=False)
                             
    for img, label in test_loader: 
        
        plt.figure(label[0])
        for i in range(9):
            plt.subplot(911+i)   
            plt.title(int(label[i]))
            # print(img) 
            # print(label)
            plt.plot(img[i])
        plt.show()
        break


generate_cifar10_iid(3)
# generate_bear_non_iid_mbnn_singe_pro(10)
#generate_mnist_non_iid_singe_pro(10)
# generate_bear_train_verif_test_dateset()
#generate_bear_non_iid_singe_pro(10)     
# generate_bear_train_test_dateset()
# generate_mnist_non_iid(worker_num=12)
# for i in range(12):
#     loader('beardata_non_iid_'+str(i)+'.npz')
# loader('data/mnist_non_iid_1.npz')
# loader('mnist_non_iid_2.npz')
# loader('mnist_non_iid_3.npz')
# loader('mnist_non_iid_4.npz')
# loader('mnist_non_iid_5.npz')
# loader('mnist_non_iid_6.npz')
# loader('mnist_non_iid_7.npz')
# loader('mnist_non_iid_8.npz')
# loader('mnist_non_iid_9.npz')
# loader('mnist_non_iid_10.npz')
# loader('mnist_non_iid_11.npz')
