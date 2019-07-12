export MODEL_DIR=gs://banknotes_1/faster_rcnn_nas_coco_2018_01_28
export PATH_TO_LOCAL_YAML_FILE=./google_cloud.yaml
export PIPELINE_CONFIG_PATH=gs://banknotes_1/training/faster_rcnn_nas_coco_cloud.config
# From tensorflow/models/research/
gcloud ml-engine jobs submit training object_detection_6 --runtime-version 1.12 --job-dir=$MODEL_DIR --packages ./dist/object_detection-0.1.tar.gz,./slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz --module-name object_detection.model_main --region europe-west1 --config $PATH_TO_LOCAL_YAML_FILE -- --model_dir=$MODEL_DIR --pipeline_config_path=$PIPELINE_CONFIG_PATH