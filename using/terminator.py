import os
import os.path
import re
from datetime import datetime
import pymssql
import datetime as dt
from lex import host_check,send_to_telegram, filedate
import progressbar

life_day_lenght=3
chat = "-1001448066127"

if host_check("scriptlab.net"):
	print("scriptlab.net - Ok")
else:
	print("scriptlab.net - Unavailable. Exit")
	exit()

if host_check("10.2.4.25"):
	print("10.2.4.25 - ok")
else:
	print("10.2.4.25 - Unavailable. Exit")
	send_to_telegram(chat,"10.2.4.25 - Unavailable. Unable to terminate records. Exit")
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
		send_to_telegram(chat,log_message)
		bar = progressbar.ProgressBar(maxval=len(files), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
		i=0
		bar.start()
		for filename in files:

			if ".jpg" in filename:
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
		send_to_telegram(chat,log_message)
		files_removed_count=0
log_message="Terminator job complete. normal exit"
print(log_message)
send_to_telegram(chat,log_message)