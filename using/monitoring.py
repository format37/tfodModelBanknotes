import cv2
import os
import numpy as np
imagesPath="images/boxed/"

files = sorted(os.listdir(imagesPath))
while True:
	if np.count_nonzero(files):
		img = cv2.imread(files[0])
		cv2.imshow("monitoring", cv2.resize(img, (640,480)))
		print(files[0])
		os.unlink(imagesPath+files[0])