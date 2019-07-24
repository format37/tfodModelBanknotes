import numpy as np
import os
import sys
import getopt
import tensorflow as tf
import datetime
sys.path.append("../..")
import cv2

sys.path.append("../../..")

from utils import label_map_util
from utils import visualization_utils as vis_util

#from natsort import natsorted, ns
import shutil

class objects_recognizer():
	
	def __init__(self):
		self.gpu	= 0
		self.NUM_CLASSES = 18
		#self.PATH_TO_CKPT	= '../../banknotes_inference_graph_v3_20904/frozen_inference_graph.pb'
		self.PATH_TO_CKPT	= '../../inference_v6_10591/frozen_inference_graph.pb'#ADD
		#self.PATH_TO_LABELS = '../training/object-detection.pbtxt'
		self.PATH_TO_LABELS = '../../training/mscoco_label_map.pbtxt'#ADD
		self.source_path	= ""
		self.save_path	= "results/"		
		
	def run(self,file_list):

		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]=str(self.gpu)

		score_limit	= 0.1
		sizeMin	= 10
		sizeMax	= 1090
		font = cv2.FONT_HERSHEY_SIMPLEX
		files_result	= []
		
		detection_graph = tf.Graph()
		with detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')
		
		label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)				
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
		category_index = label_map_util.create_category_index(categories)		
		with detection_graph.as_default():
			with tf.Session(graph=detection_graph) as sess:
				
				for file_current in file_list:
					#print(self.source_path+file_current)
					image_np = cv2.imread(self.source_path+file_current)
					#image_original	= image_np.copy()
					
					#RECOGNIZE++
					image_np_expanded=np.expand_dims(image_np, axis=0)
					image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
					boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
					scores = detection_graph.get_tensor_by_name('detection_scores:0')
					classes = detection_graph.get_tensor_by_name('detection_classes:0')
					num_detections = detection_graph.get_tensor_by_name('num_detections:0')
					(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
					final_score = np.squeeze(scores)
					
					# vis_util.visualize_boxes_and_labels_on_image_array(
						# image_np,
						# np.squeeze(boxes),
						# np.squeeze(classes).astype(np.int32),
						# np.squeeze(scores),
						# category_index,
						# use_normalized_coordinates=True,
						# line_thickness=2)
					
					#get objects detected count
					objectsDetectedCount	= 0
					score_summ	= 0
					npsboxes=np.squeeze(boxes)
					imageheight, imagewidth  = image_np.shape[:2]
																				
					for i in range(100):
						current_class_id	= np.squeeze(classes).astype(np.int32)[i]
						current_class_name	= category_index[current_class_id]['name']
						if (scores is None or final_score[i] > score_limit) and "banknotes" in current_class_name:
							
							#ensurance boxes
							rx1=int(npsboxes[i][1]*imagewidth)	#1 xLeft
							ry1=int(npsboxes[i][0]*imageheight)	#0 yTop
							rx2=int(npsboxes[i][3]*imagewidth)	#3 xRight
							ry2=int(npsboxes[i][2]*imageheight)	#2 yBottom
							
							xlen=rx2-rx1
							ylen=ry2-ry1
							
							if xlen>sizeMin and xlen<sizeMax and ylen>sizeMin and ylen<sizeMax:
								
								cv2.rectangle(image_np, (rx1,ry1) , (rx2,ry2) , (0,255,0) , 2)
								cv2.rectangle(image_np, (rx1,ry1+1) , (rx1+200,ry1-15) , (0,255,0) , -1 )
								
								object_description	= str(round(final_score[i]*100))+"% "+current_class_name
								fontScale              = 0.5								
								lineType               = 2
								cv2.putText(image_np, object_description, (rx1,ry1), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0,0,0), lineType) 
								
								score_summ	= score_summ+final_score[i]
								
								objectsDetectedCount+=1
								
					if objectsDetectedCount>0:						
						score_summString	= '%.2f'%(score_summ/objectsDetectedCount)#middle
						score_summStringH	= '%.2f'%(score_summ/objectsDetectedCount*100)#middle
					else:
						score_summString	= "0"
						score_summStringH	= "0"
					#RECOGNIZE--
					
					if objectsDetectedCount>0:						
						path_to_save=self.save_path+file_current;
						files_result.append(file_current)
						cv2.imwrite(path_to_save,image_np)
						print(datetime.datetime.now().strftime("%Y.%m.%d  %H:%M:%S")+" "+path_to_save);
					
		return files_result