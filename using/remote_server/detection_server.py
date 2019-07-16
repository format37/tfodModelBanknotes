import socket
import pickle
import datetime
from image_detection_lib import objects_recognizer

server_ip= "10.2.4.87"
server_port	= 9090

sock = socket.socket()

sock.bind(('', server_port))
sock.listen(1)
while True:
	try:
		conn, addr = sock.accept()
		while True:
			data = conn.recv(server_port)
			if not data:
				break
			
			reco				=	objects_recognizer()	
			
			data_packet			=	pickle.loads(data)
			pathes				=	data_packet[0]
			source_files		=	data_packet[1]
			
			reco.source_path	=	pathes[0]
			reco.save_path		=	pathes[1]
			
			print(datetime.datetime.now().strftime("%Y.%m.%d  %H:%M:%S")+" Start with "+str(len(source_files))+" files")
			
			recognized_files = reco.run(source_files)
			conn.send(pickle.dumps(recognized_files))
			
			print(datetime.datetime.now().strftime("%Y.%m.%d  %H:%M:%S")+" Job complete!")
			
	except KeyboardInterrupt:
		conn.close()
		raise