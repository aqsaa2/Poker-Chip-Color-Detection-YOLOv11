# Poker Chip Color Detection using YOLOv11

This repository contains the code and resources for detecting and classifying poker chip colors in real-time using a custom YOLOv11 model. The project leverages computer vision techniques to accurately identify poker chip colors under varying conditions such as lighting and chip orientations.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)

## Project Overview

This project uses YOLOv11, an advanced object detection model, to identify and classify various poker chip colors. The model was trained on a dataset of poker chip images and fine-tuned to ensure high accuracy and reliability. The system can detect multiple chip colors in real-time and output results that can be used in gaming applications, casinos, or automated systems.

## Installation

Follow these steps to set up the project locally:

1. **Install the required dependencies:**

Install Ultralytics library
Make sure you have Python 3.x and pip installed.
or simply use google colab

2. **Set up the YOLOv11 model and datasets as specified in the repository.**

## Usage
After installing dependencies, you can run the detection system.

### Training the Model
To train the model with your custom dataset, use the following command:

!yolo task=detect mode=train model=yolo11s.pt data=/content/drive/MyDrive/Poker_Chips_Detection_YOLO/data.yaml epochs=10 imgsz=640 plots=True
### Running inference:
For predicting poker chip colors in test images:

!yolo task=detect mode=predict model=runs/detect/train/weights/best.pt conf=0.25 source=/content/drive/MyDrive/Poker_Chips_Detection_YOLO/test/images save=True

## Training
The model was trained using a custom dataset containing images of various poker chips. The training process involves the following steps:

Data preprocessing and augmentation to ensure the model generalizes well.
Custom model fine-tuning with the YOLOv11 architecture.
Evaluation with validation images to assess model accuracy.
Training results such as confusion matrices and validation images are saved in the runs/detect/train/ directory.

### Example of Training Command:

!yolo task=detect mode=train model=yolo11s.pt data=/content/drive/MyDrive/Poker_Chips_Detection_YOLO/data.yaml epochs=10 imgsz=640 plots=True
## Training Results:
After training, several results will be saved in the directory:

Confusion Matrix: A visualization of the model's classification accuracy.
Results: A summary of performance metrics.
Validation Predictions: The first batch of predictions on validation data.
You can view the results using the following commands in a Jupyter notebook:

from IPython.display import Image as IPyImage

IPyImage(filename='runs/detect/train/confusion_matrix.png', width=600)
IPyImage(filename='runs/detect/train/results.png', width=600)
IPyImage(filename='runs/detect/train/val_batch0_pred.jpg', width=600)
## Inference
After training, the model can be used to make predictions on test images. The trained model will classify poker chip colors based on input images. Predictions are saved and can be reviewed as visualized output files like results.png and val_batch0_pred.jpg.

Example command for inference:


!yolo task=detect mode=predict model=runs/detect/train/weights/best.pt conf=0.25 source=/content/drive/MyDrive/Poker_Chips_Detection_YOLO/test/images save=True
## Results Visualization:
You can examine the prediction results using the following code:

import glob
import os
from IPython.display import Image as IPyImage, display

latest_folder = max(glob.glob('/content/runs/detect/predict*/'), key=os.path.getmtime)
for img in glob.glob(f'{latest_folder}/*.jpg')[:3]:
    display(IPyImage(filename=img, width=600))
    print("\n")
This will display the top 3 prediction results with poker chip color detections.

## Results
After training and inference, the results are visualized through several plots:

Confusion Matrix: Displays the classification accuracy for each chip color.
Validation Results: Visualizes how well the model performs on unseen data.
Predictions: Example images with overlaid chip color detections.

