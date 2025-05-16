# image-classification-and-object-detection

# Deep Learning Computer Vision Projects

This repository contains two computer vision implementations:
1. **CIFAR-10 Image Classification** using CNN
2. **Object Detection** using YOLOv3

## 1. CIFAR-10 Image Classification

### Features
- CNN model with BatchNorm and Dropout
- MLflow experiment tracking
- Model checkpointing
- Training visualization
- Performance metrics

### Requirements
tensorflow>=2.0
keras
mlflow
pandas
numpy
matplotlib
seaborn
scikit-learn


### Usage
```bash
python cifar10_classification.py
2. YOLOv3 Object Detection
Features
Complete YOLOv3 implementation

Darknet-53 backbone

Multi-class detection

Bounding box visualization

MLflow integration

Requirements
tensorflow>=1.0
mlflow
numpy
pillow
opencv-python
seaborn
Setup
Download weights:

bash
wget https://pjreddie.com/media/files/yolov3.weights
Prepare coco.names file

Usage
bash
python object_detection.py
Common Setup
bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
Output Examples
Classification: Accuracy plots, prediction samples

Detection: Annotated images in /output folder

Notes
Ensure MLflow server is running for tracking

Add test images for object detection
