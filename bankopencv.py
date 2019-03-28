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
	startx=900
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
	cv2.rectangle(image_np,(920,90) ,(1120,290),(0,255,0),-1)
	return image_np

#cap = cv2.VideoCapture(0)#webcam
cap = cv2.VideoCapture("rtsp://LOGIN:PASS@IP:PORT/live/main")

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

# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_NAME = 'banknotes_interference_graph'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

#NUM_CLASSES = 90
NUM_CLASSES = 1


# ## Download Model

# In[5]:

#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
#	file_name = os.path.basename(file.name)
#	if 'frozen_inference_graph.pb' in file_name:
#		tar_file.extract(file, os.getcwd())


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

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# # Detection

# In[9]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

score_limit	= .5
lasteventdate=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
f = open('enevts.csv' , 'a')

#configfile=open('shop.config')
#if (configfile.read(1)=='y'):
#	saveimage=True
#else:
#	saveimage=False

# In[10]:
#config = tf.ConfigProto(device_count = {'CPU': 0})
with detection_graph.as_default():
	#with tf.Session(graph=detection_graph,config=config) as sess:
	with tf.Session(graph=detection_graph) as sess:
		while True:
			ret, image_np = cap.read()
			#	1	2
			#	3	4
			#image_left_top		= image_np[100:300, 100:300]	#1
			#image_right_top		= image_np[100:300, 350:550]	#2
			#image_left_bottom	= image_np[260:460, 100:300]	#3
			#image_right_bottom	= image_np[260:460, 350:550]	#4
			#image_top			= np.concatenate((image_right_bottom,		image_left_bottom),		axis=1)
			#image_bottom		= np.concatenate((image_right_top,			image_left_top),		axis=1)
			#image_np			= np.concatenate((image_top,				image_bottom),			axis=0)
			
			#image_np = cv2.imread("shop.jpeg")
			image_np=shop_quad(image_np)
			image_np=remove_dish(image_np)
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
			count = 0
			score_summ=0
			
			npsboxes=np.squeeze(boxes)
			imagewidth=1024
			imageheight=1024
						
			for i in range(100):
				if scores is None or final_score[i] > score_limit:					
					xlen=npsboxes[0][1]*imagewidth-npsboxes[0][0]*imagewidth
					ylen=npsboxes[0][3]*imageheight-npsboxes[0][2]*imageheight
					if (xlen*ylen<150*150):
						count = count + 1
						score_summ	= score_summ+final_score[i]
					#else:
						#print(""+'%.2f'%xlen+"x"+'%.2f'%ylen)
					
			if (count>0):
				datenow=datetime.datetime.now()
				eventdate=datenow.strftime("%Y,%m,%d,%H,%M,%S")
				if (eventdate!=lasteventdate):
					lasteventdate=eventdate
					score_summ='%.2f'%score_summ
					print(eventdate+' - '+score_summ)
					f.write(eventdate+','+score_summ+'\n')
					
					configfile=open('shop.config')
					if (configfile.read(1)=='y'):
						#saveimage=True
					#else:
						#saveimage=False					
					#if (saveimage):
						# Visualization of the results of a detection.
						vis_util.visualize_boxes_and_labels_on_image_array(
						image_np,
						np.squeeze(boxes),
						np.squeeze(classes).astype(np.int32),
						np.squeeze(scores),
						category_index,
						use_normalized_coordinates=True,
						line_thickness=2,
						min_score_thresh=score_limit)
				
						eventdate=datenow.strftime("%Y-%m-%d_%H_%M_%S")
						cv2.imwrite("eventimages/event"+eventdate+".jpg", image_np)
			
			#cv2.imshow('object detection', cv2.resize(image_np, (1024,1024)))
			#if cv2.waitKey(25) & 0xFF == ord('q'):
			#	cv2.destroyAllWindows()
			#	break
			#break
