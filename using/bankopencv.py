import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

#sys.path.insert(0, '/home/alex/.local/lib/python3.6/site-packages/tensorflow/models/research/object_detection')
#sys.path.append('/home/alex/.local/lib/python3.6/site-packages/tensorflow/models/research/object_detection')
sys.path.append("../..")

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
import datetime
import pymssql
import requests



contentSaveDirectory	= "/var/www/html/"

conn = pymssql.connect(server='10.2.4.25', user='ICECORP\\1csystem', password='0dKasn@ms+', database='shopEvents')
cursor = conn.cursor()

score_limit	= .1
xlenMax=9200
ylenMax=9200

imagewidth=1920
imageheight=1080

#webcam
#capAddress=0

#shop1
#capAddress="rtsp://admin:V35XB3Uz@10.0.4.102:554/live/main"

#shop2
#capAddress="rtsp://admin:V35XB3Uz@10.0.4.40:554/live/main"

#ex
#capAddress="rtsp://admin:V35XB3Uz@10.2.5.164:554/Streaming/Channels/1"

#ex
#capAddress="rtsp://admin:V35XB3Uz@10.2.6.167:554/live/main"

#file
capAddress="../../altkas1.avi"

cap = cv2.VideoCapture(capAddress)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_CKPT = 'banknotes_inference_graph_v2g/frozen_inference_graph.pb'
#PATH_TO_CKPT = 'banknotes_inference_graph_v1/frozen_inference_graph.pb'
#PATH_TO_CKPT = 'banknotes_inference_graph_v3_20904/frozen_inference_graph.pb'
PATH_TO_CKPT = '../../banknotes_inference_graph_v5_5037/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')
PATH_TO_LABELS = '../training/object-detection.pbtxt'

NUM_CLASSES = 1

def rotate_bound(image, angle):
	# grab the dimensions of the image and then determine the
	# centre
	(h, w) = image.shape[:2]
	(cX, cY) = (w // 2, h // 2)

	# grab the rotation matrix (applying the negative of the
	# angle to rotate clockwise), then grab the sine and cosine
	# (i.e., the rotation components of the matrix)
	M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])

	# compute the new bounding dimensions of the image
	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))

	# adjust the rotation matrix to take into account translation
	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY

	# perform the actual rotation and return the image
	return cv2.warpAffine(image, M, (nW, nH))

def shop_quad(image_np):
	#rotate
	rotated=rotate_bound(image_np,353)
	#crop
	quadsize=640
	#[startY:endY,startX,endX]
	startx=300
	starty=950
	cropD	= rotated[starty:starty+quadsize, startx:startx+quadsize]
	startx=810
	starty=700
	cropA	= rotated[starty:starty+quadsize, startx:startx+quadsize]
	startx=1450
	starty=700
	cropB	= rotated[starty:starty+quadsize, startx:startx+quadsize]
	startx=1980
	starty=700
	cropC	= rotated[starty:starty+quadsize, startx:startx+quadsize]
	#concatenate
	image_top=np.concatenate((cropA,cropB),axis=1)
	image_bottom=np.concatenate((cropD,cropC),axis=1)
	return np.concatenate((image_top,image_bottom),axis=0)

def drawMask(image_np):
	#cv2.rectangle(image_np,(970,90) ,(1170,290),(0,255,0),-1)
	x1=0
	y1=0
	x2=x1+1420
	y2=y1+310+150
	cv2.rectangle(image_np,(x1,y1) ,(x2,y2),(0,0,0),-1)
	return image_np

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("../../..")
#sys.path.insert(0, '/home/alex/.local/lib/python3.6/site-packages/tensorflow/models/research/object_detection')
# ## Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils as vis_util

# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# ## Helper code

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# # Detection

lasteventdate=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
objects_count = 0
score_summ=0
imageSaved=False
logFile= open('log.txt' , 'a')

#config = tf.ConfigProto(device_count = {'CPU': 0})
with detection_graph.as_default():
	#with tf.Session(graph=detection_graph,config=config) as sess:
	with tf.Session(graph=detection_graph) as sess:
		while True:
			
			ret, image_np = cap.read()
			
			if ret:
				
				objectsPerTime=0							
				#configfile=open('shop.config')
				
				#image_np = cv2.imread("shop3.jpg")
				#image_np=shop_quad(image_np)
				imageSource=image_np;
				#image_np=drawMask(image_np)
				
				# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
				image_np_expanded = np.expand_dims(image_np, axis=0)
				image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
				# Each box represents a part of the image where a particular object was detected.
				boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
				# Each score represent how level of confidence for each of the objects.
				# Score is shown on the result image, together with the class label.
				scores = detection_graph.get_tensor_by_name('detection_scores:0')
				classes = detection_graph.get_tensor_by_name('detection_classes:0')
				num_detections = detection_graph.get_tensor_by_name('num_detections:0')
				# Actual detection.
				(boxes, scores, classes, num_detections) = sess.run(
				[boxes, scores, classes, num_detections],
				feed_dict={image_tensor: image_np_expanded})
				
				final_score = np.squeeze(scores)
				
				npsboxes=np.squeeze(boxes)			
				
				datenow=datetime.datetime.now()
				#eventdate=datenow.strftime("%Y,%m,%d,%H,%M,%S")
				eventdate=datenow.strftime("%Y,%m,%d,%H,%M")				
				
				current_image_objects_count = 0
				
				#Boxes EX++
				# Here output the category as string and score to terminal
				#print([category_index.get(i) for i in classes[0]])
				#print(scores)
				#boxes_shape = boxes.shape
				#if boxes_shape:
					#for i in range(boxes_shape[0]):
						#if len(boxes_shape) != 2 or boxes_shape[1] != 4:
							#raise ValueError('Input must be of size [N, 4]')
						#	print('Input must be of size [N, 4]')
						#else:
					#for i in range(boxes_shape[0]):
						#display_str_list = ()
						#if display_str_list_list:
						#display_str_list = display_str_list_list[i]
						#print("boxes[i, 0] "+'%.1i'%boxes[i, 0]);
						#, boxes[i, 1], boxes[i, 2],boxes[i, 3], color, thickness, display_str_list)
				#Boxes EX--
				
				for i in range(100):
					if scores is None or final_score[i] > score_limit:
						#0 yTop
						#1 xLeft
						#2 yBottom
						#3 xRight
						xlen=npsboxes[i][3]*imagewidth-npsboxes[i][1]*imagewidth
						ylen=npsboxes[i][2]*imageheight-npsboxes[i][0]*imageheight
						#if (xlen<xlenMax and ylen<ylenMax):
						if True:
							current_image_objects_count = current_image_objects_count + 1
							objects_count = objects_count + 1
							score_summ	= score_summ+final_score[i]
							print('%.1i'%current_image_objects_count+' of '+'%.1i'%objects_count+' i='+'%.1i'%i+' '+'%.2i'%xlen+" v "+'%.2i'%ylen)
							#print('%.1i'%current_image_objects_count+' of '+'%.1i'%objects_count+' i='+'%.1i'%i+' i0 '+'%.2f'%npsboxes[i][0]+" i1 "+'%.2f'%npsboxes[i][1]+" i2 "+'%.2f'%npsboxes[i][2]+" i3 "+'%.2f'%npsboxes[i][3])
							
							#configfile=open('shop.config')
							#if (configfile.read(1)=='y' and imageSaved==False):
							#if (imageSaved==False):
							if (True):
								# Visualization of the results of a detection.
								vis_util.visualize_boxes_and_labels_on_image_array(
								image_np,
								npsboxes,
								np.squeeze(classes).astype(np.int32),
								np.squeeze(scores),
								category_index,
								use_normalized_coordinates=True,
								line_thickness=1,
								min_score_thresh=score_limit)						
								fileEventDate=datenow.strftime("%Y-%m-%d_%H_%M")
								imageBoxed=image_np;
								toSaveDay=datenow.strftime("%Y-%m-%d")
								
								imagesBoxedDirectory	= contentSaveDirectory+"events/"+toSaveDay+"/boxed/"
								imagesSourceDirectory	= contentSaveDirectory+"events/"+toSaveDay+"/source/"
								
								if not os.path.exists(imagesBoxedDirectory):
									os.makedirs(imagesBoxedDirectory)
									
								if not os.path.exists(imagesSourceDirectory):
									os.makedirs(imagesSourceDirectory)
									
								toSaveName=fileEventDate+'_s'+'%.2i'%xlen+"x"+'%.2i'%ylen
								#toSaveSourceName=fileEventDate+'_s'+'%.2i'%xlen+"x"+'%.2i'%ylen
								
								##print('%.3i'%objects_count+' '+'%.2i'%xlen+" v "+'%.2i'%ylen+" s:"+'%.2f'%final_score[i])
								
								imageSaved=True
						#else:
							#print('%.1i'%current_image_objects_count+' of '+'%.1i'%objects_count+' i='+'%.1i'%i+' '+'%.2i'%xlen+" o "+'%.2i'%ylen+" s:"+'%.2f'%final_score[i])
				
				cv2.imshow('object detection', cv2.resize(image_np, (960,540)))
				if cv2.waitKey(25) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break
				
				if (objects_count>0):
					objectsPerTime=objectsPerTime+objects_count;		
					if (eventdate!=lasteventdate):#save objectsPerMinute
						lasteventdate=eventdate
						score_summString='%.2f'%(score_summ/objects_count)#middle
						objectsPerTimeString	= '%.1i'%objectsPerTime
						print(eventdate+' - '+score_summString+' * '+objectsPerTimeString)
						
						#save CSV
						toSaveName=toSaveName				+'_p'+score_summString+'_o'+objectsPerTimeString
						eventsFileName=contentSaveDirectory+"events/"+toSaveDay+"/"+datenow.strftime("%Y.%m.%d")+'.csv'
						eventsFile = open(eventsFileName, 'a')
						eventsFile.write(eventdate+','+'0,'+score_summString+','+objectsPerTimeString+","+toSaveName+".jpg"+'\n')
						
						#save JPG
						configfile=open('shop.config')
						if (configfile.read(1)=='y' and imageSaved==True):							
							print(toSaveName)
							boxedImagePath	= imagesBoxedDirectory	+ toSaveName+".jpg"
							cv2.imwrite(boxedImagePath, imageBoxed)
							cv2.imwrite(imagesSourceDirectory	+ toSaveName+".jpg", imageSource)
							
							#save to Telegram
							print(boxedImagePath)
							with open(boxedImagePath,'rb') as fh:
								mydata = fh.read()
								#payload = {'username': 'bob', 'email': 'bob@bob.com'}
								response = requests.put('http://scriptlab.net/telegram/bots/relaybot/relayPhotoViaPutShop.php',data=mydata,headers={'content-type':'text/plain'},params={'file': boxedImagePath})
							
						#save SQL
						cursor.execute("INSERT INTO events (eventDate,objectsCount,middleScore,FileName) VALUES ('"+datenow.strftime("%Y-%m-%dT%H:%M:%S")+"',"+objectsPerTimeString+","+score_summString+",'"+toSaveName+"')")
						conn.commit()
						
						#reset
						objects_count = 0
						objectsPerTime	= 0
						score_summ=0
						imageSaved=False
				
				
			else:
				#shop
				cap = cv2.VideoCapture(capAddress)
			
			#cv2.imshow('object detection', cv2.resize(image_np, (1024,1024)))
			#cv2.imshow('object detection', image_np)
			#if cv2.waitKey(25) & 0xFF == ord('q'):
			#	cv2.destroyAllWindows()
			#	break
			#break
conn.close()
