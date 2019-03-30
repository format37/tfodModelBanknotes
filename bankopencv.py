import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
import datetime

score_limit	= .1

#webcam
#cap = cv2.VideoCapture(0)

#file
#cap = cv2.VideoCapture("altkas1.avi")

#shop
cap = cv2.VideoCapture("rtsp://admin:V35XB3Uz@10.0.4.102:554/live/main")

#it
#cap = cv2.VideoCapture("rtsp://admin:V35XB3Uz@10.2.6.167:554/live/main")

#ex
#cap = cv2.VideoCapture("rtsp://admin:V35XB3Uz@10.2.5.164:554/Streaming/Channels/1")

#print("camera connected")

# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
#MODEL_NAME = 'banknotes_interference_graph'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_CKPT = 'banknotes_interference_graph/frozen_inference_graph.pb'
#PATH_TO_CKPT = 'banknotes_inference_graph_v2b/frozen_inference_graph.pb'
PATH_TO_CKPT = 'banknotes_inference_graph_v2c/frozen_inference_graph.pb'
#PATH_TO_CKPT =	"/home/alex/.local/lib/python3.6/site-packages/tensorflow/models/research/object_detection/banknotes_inference_graph_v2/frozen_inference_graph.pb"

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

#NUM_CLASSES = 90
NUM_CLASSES = 1

def rotate_bound(image, angle):
	# grab the dimensions of the image and then determine the
	# centre
	#try:
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
	#except AttributeError:
		#return image

def shop_quad(image_np):
	#rotate
	rotated=rotate_bound(image_np,353)
	#crop
	quadsize=640
	#[startY:endY,startX,endX]
	startx=300
	starty=950
	cropD	= rotated[starty:starty+quadsize, startx:startx+quadsize]
	#startx=900
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

def remove_dish(image_np):
	cv2.rectangle(image_np,(970,90) ,(1170,290),(0,255,0),-1)
	return image_np

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# ## Load a (frozen) Tensorflow model into memory.

# In[6]:

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

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
#IMAGE_SIZE = (12, 8)

lasteventdate=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#configfile=open('shop.config')
#if (configfile.read(1)=='y'):
#	saveimage=True
#else:
#	saveimage=False

#boxSizes=np.array([])
objects_count = 0
score_summ=0
imageSaved=False
logFile= open('log.txt' , 'a')

#config = tf.ConfigProto(device_count = {'CPU': 0})
with detection_graph.as_default():
	#with tf.Session(graph=detection_graph,config=config) as sess:
	with tf.Session(graph=detection_graph) as sess:
		while True:
			#try:
			
			#datenow=datetime.datetime.now()
			#print (datenow.strftime("%Y.%m.%d %H:%M:%S")+" read")
			#logFile.write(datenow.strftime("%Y.%m.%d %H:%M:%S")+" read")
			
			ret, image_np = cap.read()
			
			#cv2.imwrite("eventimages/x.jpg", image_np)
			if ret:
				
				
				#im_width, im_height = image_np.shape[:2]
				#print('%.1i'%im_width+'-'+'%.1i'%im_height)
				#logFile.write('%.1i'%im_width+'-'+'%.1i'%im_height)
				
				imagewidth=1024
				imageheight=1024
				objectsPerTime=0							
				configfile=open('shop.config')
				
				#image_np = cv2.imread("shop.jpeg")				
				image_np=shop_quad(image_np)
				#image_np=remove_dish(image_np)
				
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
				
				#eventdate=datenow.strftime("%Y-%m-%d_%H_%M")
				
				for i in range(100):
					if scores is None or final_score[i] > score_limit:					
						xlen=npsboxes[0][1]*imagewidth-npsboxes[0][0]*imagewidth
						ylen=npsboxes[0][3]*imageheight-npsboxes[0][2]*imageheight
						if (xlen*ylen<350*350):
						#if True:
							objects_count = objects_count + 1
							score_summ	= score_summ+final_score[i]
							#boxSizes=np.append(boxSizes, '%.2i'%xlen+" v "+'%.2i'%ylen)
							#print("count")
							print('%.2i'%objects_count+' '+'%.2i'%xlen+" v "+'%.2i'%ylen)
							
							configfile=open('shop.config')
							if (configfile.read(1)=='y' and imageSaved==False):
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
								#eventdate=datenow.strftime("%Y-%m-%d_%H_%M_%S")								
								fileEventDate=datenow.strftime("%Y-%m-%d_%H_%M")
								#filename="eventimages/event"+fileEventDate+'_c'+'%.1i'%objects_count+'s'+'%.2i'%xlen+"x"+'%.2i'%ylen+".jpg"
								#filename=
								toSaveImage=image_np;
								toSaveName="eventimages/event"+fileEventDate+'_s'+'%.2i'%xlen+"x"+'%.2i'%ylen
								#cv2.imwrite(filename, image_np)
								#print(filename)
								print('%.3i'%objects_count+' '+'%.2i'%xlen+" v "+'%.2i'%ylen)
								imageSaved=True
						else:
							#boxSizes=np.append(boxSizes, '%.2i'%xlen+" o "+'%.2i'%ylen)
							print('--- '+'%.2i'%xlen+" o "+'%.2i'%ylen)
							#print(""+'%.2f'%xlen+"x"+'%.2f'%ylen)
							
				
				
				if (objects_count>0):
					objectsPerTime=objectsPerTime+objects_count;		
					if (eventdate!=lasteventdate):#save objectsPerMinute
						lasteventdate=eventdate
						score_summString='%.2f'%(score_summ/objects_count)#middle
						objectsPerTimeString	= '%.1i'%objectsPerTime
						print(eventdate+' - '+score_summString+' * '+objectsPerTimeString)
						#for i in boxSizes:
							#print(i)
						
						#reset
						#boxSizes=np.array([])
						eventsFile = open('enevts.csv' , 'a')
						eventsFile.write(eventdate+','+'0,'+score_summString+','+objectsPerTimeString+'\n')
						configfile=open('shop.config')
						if (configfile.read(1)=='y' and imageSaved==True):
							toSaveName=toSaveName+'_p'+score_summString+'_o'+objectsPerTimeString
							print(toSaveName)
							cv2.imwrite(toSaveName+".jpg", toSaveImage)
						
						objects_count = 0
						objectsPerTime	= 0
						score_summ=0
						imageSaved=False
			else:
				
				#print (datenow.strftime("%Y.%m.%d %H:%M:%S")+" lost")
				#logFile.write(datenow.strftime("%Y.%m.%d %H:%M:%S")+" lost")
				
				#shop
				cap = cv2.VideoCapture("rtsp://admin:V35XB3Uz@10.0.4.102:554/live/main")
				
			#break
			#except:
#				print("except")
				#if ret==True:
					#video_out.write(image_np)
				#if cv2.waitKey(1) & 0xFF == ord('q'):
					#out.release()
					#break
			
			#cv2.imshow('object detection', cv2.resize(image_np, (1024,1024)))
			#if cv2.waitKey(25) & 0xFF == ord('q'):
			#	cv2.destroyAllWindows()
			#	break
			#print("the end")
			#break