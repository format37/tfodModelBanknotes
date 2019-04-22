import numpy as np
import os
import six.moves.urllib as urllib
import sys
import getopt
import tarfile
import tensorflow as tf
import zipfile
import datetime

sys.path.append("../..")

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import pymssql
import cv2
#capAddress="rtsp://admin:V35XB3Uz@10.0.4.102:554/live/main"#shop1
capAddress="rtsp://admin:V35XB3Uz@10.0.4.40:554/live/main"#shop2
sys.path.append("../../..")
debug=False

from utils import label_map_util
from utils import visualization_utils as vis_util

def readConfig(filename,result):
	configFile = open(filename,'r')
	try:
		result=configFile.read()
	finally:
		configFile.close()
	return result

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def main(argv):	
	inputfile = ''
	outputfile = ''
	try:
		opts, args = getopt.getopt(argv,"hi:g:",["interval=","gpu="])
	except getopt.GetoptError:
		print ('test.py -i <interval> -o <gpu>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print ('test.py -i <innterval> -g <gpu>')
			sys.exit()
		elif opt in ("-i", "--interval"):
			ops_interval = int(arg)
		elif opt in ("-g", "--gpu"):
			ops_gpu = int(arg)
	print ('Interval: ', ops_interval)
	print ('Gpu: ', ops_gpu)
	
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"]=str(ops_gpu)
	
	score_limit	= 0.1
	PATH_TO_CKPT = '../../banknotes_inference_graph_v3_20904/frozen_inference_graph.pb'
	PATH_TO_LABELS = '../training/object-detection.pbtxt'
	NUM_CLASSES = 1
	imagesBoxedDirectory	= "images/boxed/"
	sizeMin	= 50
	sizeMax	= 190
	if not os.path.exists(imagesBoxedDirectory):
		os.makedirs(imagesBoxedDirectory)
	initContentSaveDirectory	= "/var/www/html/"
	if debug==False:
		conn = pymssql.connect(server='10.2.4.25', user='ICECORP\\1csystem', password='0dKasn@ms+', database='shopEvents')
		cursor = conn.cursor()
	font = cv2.FONT_HERSHEY_SIMPLEX
	
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
	
	secondlast	= int(datetime.datetime.now().strftime("%S"))	
	time_offsets=np.arange(-10+ops_gpu,70,ops_interval)	
	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			while True:
				dtnow=datetime.datetime.now()
				secondcurrent	= int(dtnow.strftime("%S"))
				if secondcurrent!=secondlast:
					secondlast=secondcurrent			
					is_it_my_time=np.count_nonzero(np.where(time_offsets==secondcurrent))
					if is_it_my_time:
						cap = cv2.VideoCapture(capAddress)
						ret, image_np = cap.read()
						if ret:
							if debug:
								#image_np = cv2.imread("found.jpg")
								image_np = cv2.imread("empty.jpg")
							
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
																			
							score_limit	= float(readConfig('score_limit.config',	score_limit))
							sizeMin		= int(readConfig('sizeMin.config',			sizeMin))
							sizeMax		= int(readConfig('sizeMax.config',			sizeMax))
							
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
										objectsDetectedCount=objectsDetectedCount+1
										score_summ	= score_summ+final_score[i]
										cv2.rectangle(image_np, (rx1,ry1) , (rx2,ry2) , (0,255,0) , 1)
							
							if objectsDetectedCount>0:						
								score_summString	= '%.2f'%(score_summ/objectsDetectedCount)#middle
								score_summStringH	= '%.2f'%(score_summ/objectsDetectedCount*100)#middle
							else:
								score_summString	= "0"
								score_summStringH	= "0"
							
							#save for monitoring						
							#cv2.putText(image_np,"GPU-"+str(ops_gpu)+" score: "+score_summStringH,(10,150), font, 1,(255,255,255),1,cv2.LINE_AA)
							cv2.putText(image_np,str(objectsDetectedCount)+":"+score_summStringH+"%",(10,150), font, 1,(255,255,255),1,cv2.LINE_AA)
							#cv2.imwrite(imagesBoxedDirectory+dtnow.strftime("%Y-%m-%d_%H_%M_%S")+".jpg", image_np)
							
							#save for report
							if debug==False and objectsDetectedCount>0:
								toSaveDay=dtnow.strftime("%Y-%m-%d")
								contentSaveDirectory	= initContentSaveDirectory+"events/"+toSaveDay+"/boxed/"
								if not os.path.exists(contentSaveDirectory):
									os.makedirs(contentSaveDirectory)
								contentSaveFileName	= dtnow.strftime("%Y-%m-%d_%H_%M_%S")+"_obj_"+str(objectsDetectedCount)
								contentSavePath	= contentSaveDirectory	+ contentSaveFileName +".jpg"
								cv2.imwrite(contentSavePath, image_np)
								print(contentSaveFileName)
							
								#save to sql							
								cursor.execute("INSERT INTO events (eventDate,objectsCount,middleScore,FileName) VALUES ('"+dtnow.strftime("%Y-%m-%dT%H:%M:%S")+"',"+str(objectsDetectedCount)+","+score_summString+",'"+contentSaveFileName+"')")
								conn.commit()

							print(str(ops_gpu)+": "+dtnow.strftime("%H:%M:%S")+" "+score_summStringH+"% in "+str(objectsDetectedCount)+" objects")
							
							if debug:
								cv2.imwrite("result.jpg", image_np)
								break
						else:
							print(dtnow.strftime("%H:%M:%S")+" no image")
				
if __name__ == "__main__":
	main(sys.argv[1:])