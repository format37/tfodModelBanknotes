import os
import re
from datetime import datetime
import requests

path = '/mnt/shares/recognized_images/'
life_day_lenght=7

def send_to_telegram(message):
	chat= "-1001448066127"
	url	= "http://scriptlab.net/telegram/bots/relaybot/relaylocked.php?chat="+chat+"&text="+message
	requests.get(url)

date_current=datetime.now()
files_removed_count=0
	
send_to_telegram( "removing recognized files, older than "+str(life_day_lenght)+" days..")
	
for root, subdirs, files in os.walk(path):
	list_file_path = os.path.join(root, 'my-directory-list.txt')

	with open(list_file_path, 'wb') as list_file:

		for filename in files:

			if ".jpg" in filename:
				file_path = os.path.join(root, filename)
				time_difference=(date_current-datetime.fromtimestamp( os.path.getctime(file_path) )).total_seconds()
				day_difference=int(time_difference/60/60/24)

				if day_difference>life_day_lenght:
					#print( 'removing %s: %s            ' % (day_difference,file_path) , end='\r', flush=True)
					os.remove(file_path)
					files_removed_count+=1

				#else:
					#print('yang %s file: %s            ' % (day_difference,filename) , end='\r', flush=True)
			
			#else:
				#print('not jpeg: %s            ' % (filename) , end='\r', flush=True)
				
send_to_telegram( "removed "+str(files_removed_count)+" files")