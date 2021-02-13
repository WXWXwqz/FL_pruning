import socket

clientSocket = socket.socket()
 
clientSocket.connect(('192.168.1.119',9091))

clientSocket.send(b'I am a client')
 

recvData = clientSocket.recv(1024).decode('utf-8')
print('客户端收到服务器回复的消息:%s'%(recvData))
 

clientSocket.close()