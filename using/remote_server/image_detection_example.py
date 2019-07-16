from image_detection_lib import objects_recognizer

reco	= objects_recognizer()
source_files=[
	"image0.jpg",
	"image1.jpg"
	]
reco.source_path="./"
reco.save_path="./results/"
recognized_files = reco.run(source_files)

for filename in recognized_files:
	print(filename+"\n")