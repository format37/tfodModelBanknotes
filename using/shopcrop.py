import cv2
import numpy as np

from os import listdir
from os.path import isfile, join
#from shutil import copyfile

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
	#rotated=image_np
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
	image_bottom=np.concatenate((cropC,cropD),axis=1)
	return np.concatenate((image_top,image_bottom),axis=0)

mypath="/home/alex/.local/lib/python3.6/site-packages/tensorflow/models/research/object_detection/shots"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
i=0
for f in onlyfiles:
	image_np = cv2.imread(mypath+"/"+f)
	image_np=shop_quad(image_np)
	#if (f!='Алтуфьево касса (2019.03.19 14-44-53.116).jpeg'):
	print(f)
	cv2.imwrite('/home/alex/.local/lib/python3.6/site-packages/tensorflow/models/research/object_detection/out/image%s.jpg' % i, image_np);
	i+=1
	#break
#print(len(onlyfiles))

#image_np=shop_quad(image_np)
#cv2.imshow('shop', image_np)
#cv2.waitKey()
