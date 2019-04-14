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

import cv2
capAddress="rtsp://admin:V35XB3Uz@10.0.4.40:554/live/main"
sys.path.append("../../..")

from utils import label_map_util
from utils import visualization_utils as vis_util

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

	PATH_TO_CKPT = '../../banknotes_inference_graph_v3_20904/frozen_inference_graph.pb'
	PATH_TO_LABELS = '../training/object-detection.pbtxt'
	NUM_CLASSES = 1

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
				secondcurrent	= int(datetime.datetime.now().strftime("%S"))
				if secondcurrent!=secondlast:
					secondlast=secondcurrent			
					is_it_my_time=np.count_nonzero(np.where(time_offsets==secondcurrent))
					#print(str(secondcurrent)+"."+str(is_it_my_time))
					if is_it_my_time:
						cap = cv2.VideoCapture(capAddress)
						ret, image_np = cap.read()
						image_np_expanded=np.expand_dims(image_np, axis=0)
						image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
						boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
						scores = detection_graph.get_tensor_by_name('detection_scores:0')
						classes = detection_graph.get_tensor_by_name('detection_classes:0')
						num_detections = detection_graph.get_tensor_by_name('num_detections:0')
						(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})

						vis_util.visualize_boxes_and_labels_on_image_array(
							image_np,
							np.squeeze(boxes),
							np.squeeze(classes).astype(np.int32),
							np.squeeze(scores),
							category_index,
							use_normalized_coordinates=True,
							line_thickness=8)
						#cv2.imshow('object detection '+str(ops_gpu), cv2.resize(image_np, (640,480)))
						print(str(ops_gpu)+"."+str(secondcurrent))
							
						if cv2.waitKey(25) & 0xFF == ord('q'):
							cap.release()
							cv2.destroyAllWindows()
							break
				
if __name__ == "__main__":
	main(sys.argv[1:])