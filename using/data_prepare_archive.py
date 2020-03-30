import os
import os.path
import re
from datetime import datetime
import pymssql
import datetime as dt
from lex import host_check,send_to_telegram, filedate
import progressbar
from natsort import natsorted, ns

def terminator():
	life_day_lenght=60
	chat = "-1001448066127"

	if host_check("scriptlab.net"):
		print("scriptlab.net - Ok")
	else:
		print("scriptlab.net - Unavailable. Exit")
		exit()

	if host_check("10.2.4.95"):
		print("10.2.4.95 (images server) - ok")
	else:
		print("10.2.4.95 (images server) - Unavailable. Exit")
		send_to_telegram(chat,"10.2.4.95 (images server) - Unavailable. Unable to terminate records. Exit")
		exit()

	if host_check("10.2.4.25"):
		print("10.2.4.25 (SQL) - ok")
	else:
		print("10.2.4.25 (SQL) - Unavailable. Exit")
		send_to_telegram(chat,"10.2.4.25 (SQL) - Unavailable. Unable to terminate records. Exit")
		exit()

	date_current=datetime.now()

	#send_to_telegram(chat,"Removing old files..")

	conn = pymssql.connect(server='10.2.4.25', user='ICECORP\\1csystem', password='0dKasn@ms+', database='shopEvents')
	cursor = conn.cursor()
	time_limit	= (date_current - dt.timedelta(days=life_day_lenght)).strftime('%Y-%m-%d %H:%M:%S')

	# === Remove files with SQL
	query	= "SELECT path FROM files where date<'"+time_limit+"'"
	cursor.execute(query)
	answer=cursor.fetchall()
	count_of_files_to_delete = len(answer)
	log_message	= "SQL: Removing "+str(count_of_files_to_delete)+" files"
	print(log_message)
	send_to_telegram(chat,log_message)
	bar = progressbar.ProgressBar(maxval=count_of_files_to_delete, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	i=0
	bar.start()
	file_not_found_count=0
	file_removed_count=0
	for file_name_to_delete in answer:
		if os.path.isfile(file_name_to_delete[0]):
			os.remove(file_name_to_delete[0])
			file_removed_count=file_removed_count+1
		else:
			file_not_found_count=file_not_found_count+1
		bar.update(i+1)
		i=i+1;
	bar.finish()
	log_message	= "Removed: "+str(file_removed_count)+" files"
	print(log_message)
	if file_removed_count:
		send_to_telegram(chat,log_message)

	if file_not_found_count:
		log_message	= "Not found "+str(file_not_found_count)+" files"
		print(log_message)
		send_to_telegram(chat,log_message)

		
	# === Remove SQL reocords: files
	print("Removing SQL records: files")
	query	= "DELETE FROM files where date<'"+time_limit+"'"
	cursor.execute(query)
	conn.commit()

	# === Remove SQL records: events
	print("Removing SQL records: events")
	query	= "DELETE FROM events WHERE eventDate<'"+time_limit+"'"
	cursor.execute(query)
	conn.commit()

	# === File system: remove files
	#print("File system: Removing old files")
	file_date=filedate()
	files_removed_count=0
	shares_path="/mnt/shares/"
	for root, subdirs, files in os.walk(shares_path):
		list_file_path = os.path.join(root, 'my-directory-list.txt')
		with open(list_file_path, 'wb') as list_file:
			log_message="Processing "+str(len(files))+" files in directory: "+list_file_path
			print(log_message)
			#send_to_telegram(chat,log_message)
			bar = progressbar.ProgressBar(maxval=len(files), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
			i=0
			bar.start()
			for filename in files:

				if ".jpg" in filename:
					#print(filename)
					file_date.update(filename)
					file_path = os.path.join(root, filename)
					
					#time_difference=(date_current-datetime.fromtimestamp( os.path.getctime(file_path) )).total_seconds()
					#day_difference=int(time_difference/60/60/24)
					time_difference=date_current-file_date.dateFormat()
					day_difference=int(time_difference.total_seconds()/60/60/24)

					if day_difference>life_day_lenght:
						os.remove(file_path)
						files_removed_count+=1
					#else:
					#	print("New file "+str(time_difference)+": ("+str(day_difference)+"<="+str(life_day_lenght)+"): "+file_path)
				bar.update(i+1)
				i=i+1;
			bar.finish()
			log_message="Removed by file system: "+str(files_removed_count)+" files"
			print(log_message)
			#send_to_telegram(chat,log_message)
			files_removed_count=0
	log_message="Terminator job complete. normal exit"
	print(log_message)
	send_to_telegram(chat,log_message)
	
def separate():
	gpu_count=1

	chat = PASTE_GROUP_ID

	screenshots_path="/home/alex/storage/shop_screens/video/"
	processed_path="/home/alex/storage/shop_screens/video/processed/"
	shop_names=["Altuf","Avangard","Mar","Tag"]

	if host_check("scriptlab.net"):
		print("scriptlab.net - Ok")
	else:
		print("scriptlab.net - Unavailable. Exit")
		exit()

	if host_check("10.2.4.95"):
		print("10.2.4.95 (images server) - ok")
	else:
		print("10.2.4.95 (images server) - Unavailable. Exit")
		send_to_telegram(chat,"10.2.4.95 (images server) - Unavailable. Unable to terminate records. Exit")
		exit()

	if host_check("10.2.4.25"):
		print("10.2.4.25 (SQL) - ok")
	else:
		print("10.2.4.25 (SQL) - Unavailable. Exit")
		send_to_telegram(chat,"10.2.4.25 (SQL) - Unavailable. Unable to terminate records. Exit")
		exit()

	files_total_count=0
	print("files count..")
	#send_to_telegram(chat,"files dividing started..")
	for shop_id in range(4):
		files = os.listdir(screenshots_path+shop_names[shop_id]+"/grabs/")
		files = natsorted(files, alg=ns.PATH)
		files_total_count+=len(files)
	print(files_total_count)
	#send_to_telegram(chat,str(files_total_count)+" in queue")

	files_count=0
	gpu_id=0

	conn = pymssql.connect(server='10.2.4.25', user='ICECORP\\1csystem', password=PASTE_PASS, database='shopEvents')
	cursor = conn.cursor()

	# reset all to Not processed state
	query	= "delete from files_to_process"
	cursor.execute(query)
	conn.commit()

	for shop_id in range(4):	
		print(shop_names[shop_id])
		files = os.listdir(screenshots_path+shop_names[shop_id]+"/grabs/")
		files = natsorted(files, alg=ns.PATH)
		bar = progressbar.ProgressBar(maxval=len(files), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
		bar.start()
		for file_id in range(len(files)):
			
			bar.update(file_id+1)
			file_full_path	= screenshots_path+shop_names[shop_id]+"/grabs/"+files[file_id]
			if os.stat(file_full_path).st_size==0:#empty
				os.remove(file_full_path)
				continue
			if file_full_path.find(".jpg")==-1:#not jpg
				continue
				
			query	= "INSERT INTO files_to_process (file_id,file_full_path,shop_id,gpu_id,file_name) VALUES ("+str(files_count)+",'"+file_full_path+"',"+str(shop_id)+","+str(gpu_id)+",'"+files[file_id]+"')"
			cursor.execute(query)
			conn.commit()
			
			if (gpu_count>1):
				if files_count>files_total_count/(gpu_count) and gpu_id<gpu_count:
					print(file_full_path,shop_id,gpu_id)
					gpu_id+=1
					files_count=0			
				else:
					files_count+=1
			else:
				files_count+=1
		bar.update(len(files))	

	print("\nprepared:",files_count,"files")

	#send_to_telegram(chat,"files dividing complete")
