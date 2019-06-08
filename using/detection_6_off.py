import numpy as np
import os
import six.moves.urllib as urllib
import sys
import getopt
import tarfile
import tensorflow as tf
import zipfile
import datetime
from time import time
import requests
import copy

sys.path.append("../..")

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import pymssql
import cv2

sys.path.append("../../..")

from utils import label_map_util
from utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
	
def send_to_telegram(message):
	chat= "-1001448066127"
	url	= "http://scriptlab.net/telegram/bots/relaybot/relaylocked.php?chat="+chat+"&text="+message
	requests.get(url)

class filedate:
	def __init__(self):
		self.year	= "0"
		self.month	= "0"
		self.day	= "0"
		self.hour	= "0"
		self.minute	= "0"
		self.second	= "0"
	def update(self,filename):
		self.year	= filename[0:4]
		self.month	= filename[5:7]
		self.day	= filename[8:10]
		self.hour	= filename[11:13]
		self.minute	= filename[14:16]
		self.second	= filename[17:19]
	def sqlFormat(self):
		return self.year+"-"+self.month+"-"+self.day+"T"+self.hour+":"+self.minute+":"+self.second

def main(argv):

	ops_shop_id=""
	ops_gpu=0
	#ops_date=""

	try:
		opts, args = getopt.getopt(argv,"hd:g:s:",["interval=","gpu=","shop_id="])
	except getopt.GetoptError:
		print ('detection_6_off.py -d 2019-05-22 -g 0 -s 0')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print ('detection_6_off.py -d 2019-05-22 -g 0 -s 0')
			sys.exit()
		elif opt in ("-d", "--date"):
			ops_date = arg
		elif opt in ("-g", "--gpu"):
			ops_gpu = int(arg)
		elif opt in ("-s", "--shop_id"):
			ops_shop_id = int(arg)
	#print ('Interval: ', ops_interval)
	print ('Gpu: ', ops_gpu)

	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"]=str(ops_gpu)

	screenshots_path="/mnt/shares/shop_screens/"
	shop_names=["Altuf","Avangard","Mar","Tag"]
	NUM_CLASSES = 1
	PATH_TO_CKPT = '../../banknotes_inference_graph_v3_20904/frozen_inference_graph.pb'
	PATH_TO_LABELS = '../training/object-detection.pbtxt'	
	save_images_path	= "/mnt/shares/recognized_images/"
	score_limit	= 0.1
	sizeMin	= 10
	sizeMax	= 1090
	font = cv2.FONT_HERSHEY_SIMPLEX
	
	conn = pymssql.connect(server='10.2.4.25', user='ICECORP\\1csystem', password='0dKasn@ms+', database='shopEvents')
	cursor = conn.cursor()
	
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')
	
	label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)
	
	file_date=filedate()
	
	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
	
			#step in every shop
			for shop_id in range(4):
				if ops_shop_id!="" and shop_id!=int(ops_shop_id):
					continue
				#dtnow=datetime.datetime.now()
				#today=dtnow.strftime("%Y-%m-%d_08-56")
				files = os.listdir(screenshots_path+shop_names[shop_id]+"/grabs/")
				#images = list(filter(lambda x: today in x, files))
				
				if len(files)>0:
					send_to_telegram(str(len(files))+" "+shop_names[shop_id]+" begin")
				
				detectedImagesCount	= 0
				
				for file_id in range(len(files)):
					
					file_full_path	= screenshots_path+shop_names[shop_id]+"/grabs/"+files[file_id]
					
					if os.stat(file_full_path).st_size==0:
						#drop file
						os.remove(file_full_path)
						print(files[file_id],"removed as empty")
						continue
					if file_full_path.find(".jpg")==-1:
						print(files[file_id],"skipped as not jpg file")
						continue
					#file_id=0
					print(shop_id,"recognizing:",files[file_id],file_id,"of",len(files))
					file_date.update(files[file_id])
					#print(file_date.sqlFormat())
					#print("y",file_date.year)
					#print("date extracted from filename")
					#exit()
					
					image_np = cv2.imread(file_full_path)
					#image_original	= image_np
					image_original	= image_np.copy()
					#image_original	= copy.deepcopy(image_np)
					
					#RECOGNIZE++
					image_np_expanded=np.expand_dims(image_np, axis=0)
					image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
					boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
					scores = detection_graph.get_tensor_by_name('detection_scores:0')
					classes = detection_graph.get_tensor_by_name('detection_classes:0')
					num_detections = detection_graph.get_tensor_by_name('num_detections:0')
					(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
					final_score = np.squeeze(scores)
					
					vis_util.visualize_boxes_and_labels_on_image_array(
						image_np,
						np.squeeze(boxes),
						np.squeeze(classes).astype(np.int32),
						np.squeeze(scores),
						category_index,
						use_normalized_coordinates=True,
						line_thickness=2)
					
					#get objects detected count
					objectsDetectedCount	= 0
					score_summ	= 0
					npsboxes=np.squeeze(boxes)
					imageheight, imagewidth  = image_np.shape[:2]
																	
					#score_limit	= float(readConfig('score_limit.config',	score_limit))
					#sizeMin		= int(readConfig('sizeMin.config',			sizeMin))
					#sizeMax		= int(readConfig('sizeMax.config',			sizeMax))
					
					for i in range(100):
						if scores is None or final_score[i] > score_limit:
							#ensurance boxes
							rx1=int(npsboxes[i][1]*imagewidth)	#1 xLeft
							ry1=int(npsboxes[i][0]*imageheight)	#0 yTop
							rx2=int(npsboxes[i][3]*imagewidth)	#3 xRight
							ry2=int(npsboxes[i][2]*imageheight)	#2 yBottom
							
							xlen=rx2-rx1
							ylen=ry2-ry1
							if xlen>sizeMin and xlen<sizeMax and ylen>sizeMin and ylen<sizeMax:
								print(str(i)+": "+str(xlen)+" x "+str(ylen)+" score: "+str(final_score[i]))								
								
								file_name	= files[file_id].replace(".jpg","")								
								#--- save to sql fields:
								#eventDate			
								#objectsCount		same the box_id
								#middleScore		current score
								#FileName			all boxes
								file_name_all_boxes	= file_name+"_all_boxes.jpg"
								save_path_all_boxes	= save_images_path+str(shop_id)+"/original/"+file_name_all_boxes
								#shop_id
								#box_id
								#box_left
								#box_right
								#box_top
								#box_bottom
								#filename_original	original image
								file_name_original	= file_name+".jpg"
								save_path_original	= save_images_path+str(shop_id)+"/original/"+file_name_original
								#filename_box		current box
								file_name_box	= file_name+"_"+str(objectsDetectedCount)+".jpg"
								save_path_box	= save_images_path+str(shop_id)+"/boxed/"+file_name_box

								query	= "INSERT INTO events (eventDate,objectsCount,middleScore,FileName,shop_id,box_id,box_left,box_right,box_top,box_bottom,filename_original,filename_box) VALUES ('"+file_date.sqlFormat()+"',1,"+str(final_score[i])+",'"+file_name_all_boxes+"',"+str(shop_id)+","+str(objectsDetectedCount)+","+str(rx1)+","+str(rx2)+","+str(ry1)+","+str(ry2)+",'"+file_name_original+"','"+file_name_box+"')"
								#print(query)
								#exit()
								cursor.execute(query)
								conn.commit()
								
								score_summ	= score_summ+final_score[i]
								#image_np_current_box	= image_original
								image_np_current_box	= image_original.copy()
								#image_np_current_box	= copy.deepcopy(image_original)
								cv2.rectangle(image_np_current_box, (rx1,ry1) , (rx2,ry2) , (255,0,0) , 2)
								
								#save ONE BOX								
								cv2.imwrite(save_path_box, image_np_current_box)
								
								objectsDetectedCount+=1
								
					if objectsDetectedCount>0:						
						score_summString	= '%.2f'%(score_summ/objectsDetectedCount)#middle
						score_summStringH	= '%.2f'%(score_summ/objectsDetectedCount*100)#middle
					else:
						score_summString	= "0"
						score_summStringH	= "0"
					
					cv2.putText(image_np,str(objectsDetectedCount)+":"+score_summStringH+"%",(10,150), font, 1,(255,255,255),1,cv2.LINE_AA)
					#RECOGNIZE--
					
					if objectsDetectedCount>0:						
						#save ORIGINAL
						cv2.imwrite(save_path_original, image_original)
						#save ALL BOXES
						cv2.imwrite(save_path_all_boxes, image_np)
						detectedImagesCount+=1

					#drop source file
					os.remove(file_full_path)
					
				send_to_telegram(str(detectedImagesCount)+" detected in "+shop_names[shop_id])	
				
			print("job complete. normal exit")
			send_to_telegram("job complete. normal exit")
	
if __name__ == "__main__":
	main(sys.argv[1:])