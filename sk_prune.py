import socket
import fl_func
import time
import torch
class sk_prune_client():
    def __init__(self,ip='127.0.0.0',ports=9000):
        self.ip=ip
        self.ports=ports

    def prune(self,msg):
        self.clientSocket = socket.socket()  
        self.clientSocket.connect((self.ip,self.ports))
        self.clientSocket.send(('prune'+str(msg)).encode())
        print('send to prune')
        recvData = self.clientSocket.recv(1024).decode('utf-8')
        print('revcdata:%s'%(recvData))
class sk_prune_sever():
    def __init__(self,ip='127.0.0.0',ports=9000):
        self.ip=ip
        self.ports=ports 
        self.server_socket = socket.socket()
        self.server_socket.bind((self.ip,self.ports))
    def start_prune_listing(self):

        
        print('server_socket is listening ',self.ports)
        self.server_socket.listen()
        self.clientSockt,self.addr =  self.server_socket.accept() 
        data = self.clientSockt.recv(1024).decode('utf-8')
        print('sever get :%s'%(data))
        model,index = fl_func.prune_network()
        f = open('./'+data+'.txt','w')
        f.write(str(index))
        f.close()
        torch.save(model.state_dict(), './'+data+'.pkl')
        
        
        self.clientSockt.send(b'prune ok')
        print('prune ok')
 

# clientSockt.close()    #客户端对象
# server_socket.close()  #服务端对象