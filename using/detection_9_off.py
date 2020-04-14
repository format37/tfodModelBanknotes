#!/usr/bin/env python3

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import getopt
import tarfile
import tensorflow as tf
import zipfile
import datetime
import time
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

from lex import host_check,send_to_telegram, filedate
from data_prepare import terminator,separate

from natsort import natsorted, ns
import shutil

#chat = "-1001448066127"
chat = "-227727734"

if host_check("scriptlab.net"):
	print("scriptlab.net - Ok")
else:
	print("scriptlab.net - Unavailable. Exit")
	exit()

if host_check("10.2.4.95"):
	print("10.2.4.95 (images server) - ok")
else:
	print("10.2.4.95 (images server) - Unavailable. Exit")
	send_to_telegram(chat,"10.2.4.95 (images server) - Unavailable. Unable to add records. Exit")
	exit()

if host_check("10.2.4.124"):
	print("10.2.4.124 (SQL) - ok")
else:
	print("10.2.4.124 (SQL) - Unavailable. Exit")
	send_to_telegram(chat,"10.2.4.124 (SQL) - Unavailable. Unable to add records. Exit")
	exit()

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def main(argv):

	ops_shop_id=""
	ops_gpu=0

	try:
		opts, args = getopt.getopt(argv,"hd:g:s:f:t",["interval=","gpu=","shop_id=","from=","to="])
	except getopt.GetoptError:
		print ('detection_9_off.py -d 2019-05-22 -g 0 -s 0 -f 0 -t 100')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print ('detection_9_off.py -d 2019-05-22 -g 0 -s 0 -f 0 -t 100')
			sys.exit()
		elif opt in ("-d", "--date"):
			ops_date = arg
		elif opt in ("-g", "--gpu"):
			ops_gpu = int(arg)
		elif opt in ("-s", "--shop_id"):
			ops_shop_id = int(arg)
		elif opt in ("-f", "--from"):
			ops_from = int(arg)
		elif opt in ("-t", "--to"):
			ops_from = int(arg)

	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"]=str(ops_gpu)

	screenshots_path	= "/home/alex/storage/shop_screens/video/"
	processed_path		= "/home/alex/storage/shop_screens/video/processed/"
	save_images_path	= "/home/alex/storage/rcimages/"
	shop_names			= ["Altuf","Avangard","Mar","Tag","Sklad","SkladSM1","SkladSM2"]
	NUM_CLASSES 		= 18
	PATH_TO_CKPT		= '../../inference_v7_12097/frozen_inference_graph.pb'
	PATH_TO_LABELS		= '../training/mscoco_label_map.pbtxt'
	
	score_limit	= 0.7
	sizeMin	= 10
	sizeMax	= 1090
	font = cv2.FONT_HERSHEY_SIMPLEX

	#sleep_to_evening()
	prepare_data()

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
	with open('sql.pass','r') as sql_pass_file:
		sql_pass=sql_pass_file.read()
		
	with detection_graph.as_default():		
		with tf.Session(graph=detection_graph) as sess:
			
			conn = pymssql.connect(server='10.2.4.124', user='ICECORP\\1csystem', password=sql_pass, database='shopEvents')
			cursor = conn.cursor()

			while (True):				

				detectedImagesCount	= 0
				
				query	= "SELECT file_full_path,shop_id,file_name FROM files_to_process where gpu_id = "+str(ops_gpu)+" ORDER BY file_id"
				cursor.execute(query)
				sql_answer=cursor.fetchall()

				send_to_telegram(chat,"detection "+str(len(sql_answer))+" files started")
				print("len(sql_answer)",len(sql_answer))
				for record_current in sql_answer:
					
					file_full_path	= record_current[0]
					shop_id			= record_current[1]
					file_name_sql	= record_current[2]
					
					file_sql_query	= ""
					
					if not os.path.isfile(file_full_path):
						continue
					
					file_date.update(file_name_sql)
					
					image_np = cv2.imread(file_full_path)
					image_original	= image_np.copy()
					
					#RECOGNIZE++
					image_np_expanded=np.expand_dims(image_np, axis=0)
					image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
					boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
					scores = detection_graph.get_tensor_by_name('detection_scores:0')
					classes = detection_graph.get_tensor_by_name('detection_classes:0')
					num_detections = detection_graph.get_tensor_by_name('num_detections:0')
					(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
					final_score = np.squeeze(scores)
					
					#get objects detected count
					objectsDetectedCount	= 0
					score_summ	= 0
					npsboxes=np.squeeze(boxes)
					imageheight, imagewidth  = image_np.shape[:2]
																						
					for i in range(100):
						
						current_class_id	= np.squeeze(classes).astype(np.int32)[i]
						current_class_name	= category_index[current_class_id]['name']
						
						if (scores is None or final_score[i] > score_limit) and "banknotes" in current_class_name:
							
							
							rx1=int(npsboxes[i][1]*imagewidth)	#1 xLeft
							ry1=int(npsboxes[i][0]*imageheight)	#0 yTop
							rx2=int(npsboxes[i][3]*imagewidth)	#3 xRight
							ry2=int(npsboxes[i][2]*imageheight)	#2 yBottom
							
							xlen=rx2-rx1
							ylen=ry2-ry1
							
							if xlen>sizeMin and xlen<sizeMax and ylen>sizeMin and ylen<sizeMax:								
								
								file_name	= file_name_sql.replace(".jpg","")
								
								file_name_all_boxes	= file_name+"_all_boxes.jpg"
								save_path_all_boxes	= save_images_path+str(shop_id)+"/original/"+file_name_all_boxes
								file_name_original	= file_name+".jpg"
								save_path_original	= save_images_path+str(shop_id)+"/original/"+file_name_original
								file_name_box	= file_name+"_"+str(objectsDetectedCount)+".jpg"
								save_path_box	= save_images_path+str(shop_id)+"/boxed/"+file_name_box
								
								file_sql_query	= file_sql_query + "INSERT INTO files (path,date,shop_id) VALUES ('"+save_path_box+"','"+file_date.sqlFormat()+"',"+str(shop_id)+");"
								query	= "INSERT INTO events (eventDate,objectsCount,middleScore,FileName,shop_id,box_id,box_left,box_right,box_top,box_bottom,filename_original,filename_box,file_source_path) VALUES ('"+file_date.sqlFormat()+"',1,"+str(final_score[i])+",'"+file_name_all_boxes+"',"+str(shop_id)+","+str(objectsDetectedCount)+","+str(rx1)+","+str(rx2)+","+str(ry1)+","+str(ry2)+",'"+file_name_original+"','"+file_name_box+"','"+file_full_path+"')"
								cursor.execute(query)
								conn.commit()								
								score_summ	= score_summ+final_score[i]
								image_np_current_box	= image_original.copy()
								
								object_description	= str(round(final_score[i]*100))+"%"
								fontScale              = 0.5								
								lineType               = 2
								
								#boxes
								cv2.rectangle(image_np_current_box,	(rx1,ry1) 	, (rx2,ry2)			, (255,0,0) , 2)	#regular box
								cv2.rectangle(image_np_current_box,	(rx1,ry1+1)	, (rx1+200,ry1-15)	, (0,255,0) , -1)	#text background
								cv2.rectangle(image_np,				(rx1,ry1)	, (rx2,ry2)			, (0,255,0) , 2)	#regular box
								cv2.rectangle(image_np,				(rx1,ry1+1)	, (rx1+200,ry1-15)	, (0,255,0) , -1)	#text background								
								
								cv2.putText(image_np_current_box,	object_description, (rx1,ry1), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0,0,0), lineType)
								cv2.putText(image_np,				object_description, (rx1,ry1), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0,0,0), lineType)
								
								#save ONE BOX
								cv2.imwrite(save_path_box, image_np_current_box)
								print("cv2.imwrite image_np_current_box:",save_path_box)
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
						#save ALL BOXES
						cv2.imwrite(save_path_all_boxes, image_np)
						print("cv2.imwrite image_np:",save_path_all_boxes)
						detectedImagesCount+=1

						file_sql_query	= file_sql_query + "INSERT INTO files (path,date,shop_id) VALUES ('"+save_path_all_boxes+"','"+file_date.sqlFormat()+"',"+str(shop_id)+");"
						
					file_full_path_new	= processed_path+shop_names[shop_id]+"_"+file_name_sql
					os.rename(file_full_path,file_full_path_new)
					print("os.rename From:",file_full_path,"To:",file_full_path_new)
					
					# == Save filename to sql					
					file_sql_query = file_sql_query + "INSERT INTO files (path,date,shop_id) VALUES ('"+file_full_path_new+"','"+file_date.sqlFormat()+"',"+str(shop_id)+");"
					cursor.execute(file_sql_query)
					conn.commit()
					
					#remove record from file_to_process
					file_to_process_query	= "DELETE FROM files_to_process where file_full_path='"+file_full_path+"'"
					cursor.execute(file_to_process_query)
					conn.commit()
				
				sleep_to_evening()
				prepare_data()

def sleep_to_evening():
	print("calculating sleep time..")
	task_year	= (datetime.datetime.now() + datetime.timedelta(days=1)).year		
	task_month	= (datetime.datetime.now() + datetime.timedelta(days=1)).month
	task_day	= datetime.datetime.now().day
	task_hour	= 21
	task_minute	= 0
	sleeptime	= datetime.datetime(task_year, task_month, task_day, task_hour, task_minute)-datetime.datetime.now()
	print("Sleeping",sleeptime.seconds/60/60," hours")
	send_to_telegram(chat,"Sleeping "+str(sleeptime.seconds/60/60)+" hours")
	time.sleep(sleeptime.seconds)
	print("job started..")
	send_to_telegram(chat,"job started..")
		
def prepare_data():
	print("terminator..")
	terminator()
	print("separate..")
	separate()
	
if __name__ == "__main__":
	main(sys.argv[1:])
