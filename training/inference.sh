#From tensorflow/models/research/
#export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path /home/alex/.local/lib/python3.6/site-packages/tensorflow/models/research/object_detection/tfodModelBanknotes/training/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix /home/alex/.local/lib/python3.6/site-packages/tensorflow/models/research/object_detection/training_v5_5037/model.ckpt-5037 --output_directory /home/alex/.local/lib/python3.6/site-packages/tensorflow/models/research/object_detection/banknotes_inference_graph_v5_5037
