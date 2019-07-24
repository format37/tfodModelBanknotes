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
#import requests
import copy
import progressbar

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
#from sql_check_file_exist import file_exist_check

from natsort import natsorted, ns
import shutil

chat = "-1001448066127"

if host_check("scriptlab.net"):
	print("scriptlab.net - Ok")
else:
	print("scriptlab.net - Unavailable. Exit")
	exit()

if host_check("10.2.4.85"):
	print("10.2.4.85 (source images) - ok")
else:
	print("10.2.4.85 (source images) - Unavailable. Exit")
	send_to_telegram(chat,"10.2.4.85 (source images) - Unavailable. Unable to terminate records. Exit")
	exit()

if host_check("10.2.5.191"):
	print("10.2.5.191 (destination path) - ok")
else:
	print("10.2.5.191 (destination path) - Unavailable. Exit")
	send_to_telegram(chat,"10.2.5.191 (destination path) - Unavailable. Unable to terminate records. Exit")
	exit()


def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
	
#def send_to_telegram(message):
#	chat= "-1001448066127"
#	url	= "http://scriptlab.net/telegram/bots/relaybot/relaylocked.php?chat="+chat+"&text="+message
#	requests.get(url)

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
	processed_path="/mnt/shares/shop_screens/processed/"
	shop_names=["Altuf","Avangard","Mar","Tag"]
	#NUM_CLASSES = 1
	NUM_CLASSES = 18
	#PATH_TO_CKPT = '../../banknotes_inference_graph_v3_20904/frozen_inference_graph.pb'
	PATH_TO_CKPT	= '../../inference_v6_10591/frozen_inference_graph.pb'
	#PATH_TO_LABELS = '../training/object-detection.pbtxt'
	PATH_TO_LABELS = '../training/mscoco_label_map.pbtxt'
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
	
	# ==== Temporary code
	# query	= "SELECT path FROM files where path<>'' ORDER BY path"
	# cursor.execute(query)
	# sql_answer=cursor.fetchall()
	# # == prepare array
	# files_from_sql=[]
	# for record_current in sql_answer:
		# files_from_sql.append(record_current[0])
	# print("Recognized files array size: "+str(len(files_from_sql)))
	
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
					send_to_telegram(chat,str(len(files))+" "+shop_names[shop_id]+" begin")
				
				detectedImagesCount	= 0
				
				bar = progressbar.ProgressBar(maxval=len(files), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
				bar.start()
				#files=files.sort()
				files = natsorted(files, alg=ns.PATH)
				print("Processing "+str(len(files))+" files of "+shop_names[shop_id])
				for file_id in range(len(files)):
					
					bar.update(file_id+1)
					
					file_sql_query	= ""
					
					file_full_path	= screenshots_path+shop_names[shop_id]+"/grabs/"+files[file_id]
					
					if os.stat(file_full_path).st_size==0:
						#drop file
						os.remove(file_full_path)
						#print(files[file_id],"removed as empty")
						continue
					if file_full_path.find(".jpg")==-1:
						#print(files[file_id],"skipped as not jpg file")
						continue
					
					file_date.update(files[file_id])
					
					# Temporary code
					# if file_full_path in files_from_sql:
						# file_full_path_new	= processed_path+shop_names[shop_id]+"_"+files[file_id]
						# os.rename(file_full_path,file_full_path_new)
						# move_sql_query = "INSERT INTO files (path,date,shop_id) VALUES ('"+file_full_path_new+"','"+file_date.sqlFormat()+"',"+str(shop_id)+");"
						# cursor.execute(move_sql_query)
						# conn.commit()
						# continue
						
					# ============ print(shop_id,"recognizing:",files[file_id],file_id,"of",len(files))
					
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
								
								file_name	= files[file_id].replace(".jpg","")								
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
								
								object_description	= str(round(final_score[i]*100))+"% "+current_class_name
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
						file_sql_query	= file_sql_query + "INSERT INTO files (path,date,shop_id) VALUES ('"+save_path_original+"','"+file_date.sqlFormat()+"',"+str(shop_id)+");"
						file_sql_query	= file_sql_query + "INSERT INTO files (path,date,shop_id) VALUES ('"+save_path_all_boxes+"','"+file_date.sqlFormat()+"',"+str(shop_id)+");"

					#drop source file
					#os.remove(file_full_path) # <============== move
					file_full_path_new	= processed_path+shop_names[shop_id]+"_"+files[file_id]
					os.rename(file_full_path,file_full_path_new)
					
					# == Save filename to sql					
					move_sql_query = "INSERT INTO files (path,date,shop_id) VALUES ('"+file_full_path_new+"','"+file_date.sqlFormat()+"',"+str(shop_id)+");"
					cursor.execute(move_sql_query)
					conn.commit()
				
				bar.finish()				
				send_to_telegram(chat,str(detectedImagesCount)+" detected in "+shop_names[shop_id])	
				
			print("job complete. normal exit")
			send_to_telegram(chat,"job complete. normal exit")
	
if __name__ == "__main__":
	main(sys.argv[1:])