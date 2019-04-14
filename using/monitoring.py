import cv2
import os
import numpy as np
imagesPath="images/boxed/"
cv2.namedWindow('monitoring', cv2.WINDOW_NORMAL)

while True:
	files = sorted(os.listdir(imagesPath))
	if np.count_nonzero(files)>2:
		print(imagesPath+files[0])
		img = cv2.imread(imagesPath+files[0])
		cv2.imshow("monitoring", img)
		os.unlink(imagesPath+files[0])
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break