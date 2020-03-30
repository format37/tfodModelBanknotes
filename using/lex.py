import os
import requests
from datetime import datetime

def host_check(hostname):
	return True if os.system("ping -c 1 " + hostname)==0 else False
	
def send_to_telegram(chat,message):
	headers = {
    "Origin": "http://scriptlab.net",
    "Referer": "http://scriptlab.net/telegram/bots/relaybot/",
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36'}
	
	url	= "http://scriptlab.net/telegram/bots/relaybot/relaylocked.php?chat="+chat+"&text="+message
	requests.get(url,headers = headers)
	
class filedate:
	def __init__(self):
		self.year	= "0"
		self.month	= "0"
		self.day	= "0"
		self.hour	= "0"
		self.minute	= "0"
		self.second	= "0"
	def update(self,filename):
		filename=(filename[filename.find("2") : ])# for y2k only
		self.year	= filename[0:4]
		self.month	= filename[5:7]
		self.day	= filename[8:10]
		self.hour	= filename[11:13]
		self.minute	= filename[14:16]
		self.second	= filename[17:19]
	def sqlFormat(self):
		return self.year+"-"+self.month+"-"+self.day+"T"+self.hour+":"+self.minute+":"+self.second
	def dateFormat(self):
		#return datetime.strptime(self.year+"."+self.month+"."+self.day+" "+self.hour+":"+self.minute+":"+self.second,'%Y.%M.%d %H:%m:%S')
		return datetime(int(self.year), int(self.month), int(self.day), int(self.hour), int(self.minute), int(self.second))
