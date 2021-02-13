import socket

server_socket = socket.socket()
server_socket.bind(('192.168.1.119',9091))
print('server_socket is listening 9091')
server_socket.listen()

clientSockt,addr =  server_socket.accept()   #返回客户端socket通信对象和客户端的ip

data = clientSockt.recv(1024).decode('utf-8')
print('服务端收到客户端发来的消息:%s'%(data))

clientSockt.send(b'I am a server')
 

clientSockt.close()    #客户端对象
server_socket.close()  #服务端对象