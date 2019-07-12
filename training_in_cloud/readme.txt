gs://banknotes_1/data/test.record
gs://banknotes_1/data/train.record

gs://banknotes_1/training/faster_rcnn_nas_coco_cloud.config
gs://banknotes_1/training/object-detection.pbtxt

Unpacked model from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
gs://banknotes_1/faster_rcnn_nas_coco_2018_01_28

local files:
(stored in /home/alex/.local/lib/python3.6/site-packages/tensorflow/models/research)
train_cloud.sh
google_cloud.yaml

Run in terminal:
train_cloud.sh

Manual:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_cloud.md#packaging