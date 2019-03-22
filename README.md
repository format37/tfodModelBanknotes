# tensorflow object detection banknotes model
https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/

# ===
# Installation
# ===

# 1. protobuf 3.3
curl -OL https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip
unzip protoc-3.3.0-linux-x86_64.zip -d protoc3
sudo mv protoc3/bin/* /usr/local/bin/
sudo mv protoc3/include/* /usr/local/include/
sudo chown $USER /usr/local/bin/protoc
sudo chown -R $USER /usr/local/include/google

# 2. python-tk
sudo apt-get update
sudo apt-get install python-tk

# 3. Pillow 1.0
sudo apt install python3-pil # for python 3.X including python3.6

# 4. lxml
sudo apt-get install libxml2-dev libxslt1-dev python-dev

# 5. tf Slim
sudo apt-get install slim

# 6. Jupyter notebook
sudo apt install python3-notebook jupyter-core python-ipykernel  

# 7. Matplotlib
apt-cache search python3-matplotlib

# 8. Tensorflow (>=1.12.0)
pip3 install --user --upgrade tensorflow-gpu==1.13.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/
conda install cudatoolkit
conda install cudnn
python3 -c 'import tensorflow as tf; print(tf.__version__)' #work in base environment next time

# 9. Cython
sudo apt-get install cython

# 10. contextlib2
sudo apt-get install python-contextlib2

# 11. cocoapi
pip3 install pycocotools

# 12. Add Libraries to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# 13. clone Object detection git
git clone https://github.com/tensorflow/models.git

# 14. Testing the Installation
#from models/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH=/home/alex/projects/objectdetection/models/research/
export PYTHONPATH=$PYTHONPATH=/home/alex/projects/objectdetection/models/research/slim/
source ~/.bashrc
protoc object_detection/protos/*.proto --python_out=.
python object_detection/builders/model_builder_test.py

# 15. install anaconda environment tensorflow
conda install -c anaconda tensorflow-gpu
