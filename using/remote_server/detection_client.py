import socket
import pickle

#init
pathes=[
	"/mnt/shares/photo_source/",	# \\vagon-ws188\train_set\server_images\source
	"/mnt/shares/photo_result/"		# \\vagon-ws188\train_set\server_images\result
	]
source_files=[
	"image0.jpg",
	"image1.jpg"
	]
data_packet	= [pathes,source_files]

#send to remote server
server_ip= "10.2.4.87"
server_port	= 9090
sock = socket.socket()
sock.connect((server_ip, server_port))
sock.send(pickle.dumps(data_packet,2))
data = sock.recv(server_port)
sock.close()
result_list=pickle.loads(data)

#get result
for filename in result_list:
	print(filename+"\n")