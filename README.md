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


Copy code
pip install -r requirements.txt
Make sure you have Python 3.x and pip installed.

2. **Set up the YOLOv11 model and datasets as specified in the repository.**

Usage
After installing dependencies, you can run the detection system.

Training the Model
To train the model with your custom dataset, use the following command:

bash
Copy code
!yolo task=detect mode=train model=yolo11s.pt data=/content/drive/MyDrive/Poker_Chips_Detection_YOLO/data.yaml epochs=10 imgsz=640 plots=True
Running inference:
For predicting poker chip colors in test images:

bash
Copy code
!yolo task=detect mode=predict model=runs/detect/train/weights/best.pt conf=0.25 source=/content/drive/MyDrive/Poker_Chips_Detection_YOLO/test/images save=True

