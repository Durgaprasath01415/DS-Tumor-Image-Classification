# Brain Tumor MRI Image Classification

## Skills used for this project

  ●	Deep Learning

  ●	Python

  ●	TensorFlow/Keras or PyTorch

  ●	Data Preprocessing

  ●	Model Evaluation

  ●	Streamlit Deployment

## Problem Statement

This project aims to develop a deep learning-based solution for classifying brain MRI images into multiple categories according to tumor type. It involves building a custom CNN model from scratch and enhancing performance through transfer learning using pretrained models.

## 📌 Project Workflow:

1.	Understand the Dataset

    ●	Review the number of categories (tumor types) and sample images.

    ●	Check for class imbalance and image resolution consistency.

2.	Data Preprocessing

    ●	Normalize pixel values to a 0–1 range.

    ●	Resize images to a consistent shape suitable for model input (e.g. 224x224 pixels).

3.	Data Augmentation

    ●	Apply transformations like rotation, horizontal/vertical flipping, zoom, brightness adjustments, and shifts to artificially increase training data and improve model generalization

4.	Model Building

    ●	Custom CNN: Design a convolutional neural network from scratch, selecting appropriate convolution, pooling, and dense layers.

    ●	Implement dropout and batch normalization layers to avoid overfitting and stabilize learning.

5.	Transfer Learning

    ●	Load pretrained models (Example : ResNet50, MobileNetV2, EfficientNetB0).

6.	Model Training

    ●	Train both custom CNN and transfer learning models.

    ●	Track training and validation metrics.

7.	Model Evaluation

    ●	Evaluate models using metrics like accuracy, precision, recall, F1-score, and confusion matrix.

    ●	Visualize model performance trends using training history plots for accuracy and loss.

8.	Model Comparison

    ●	Compare results of custom CNN vs pretrained models.

    ●	Identify the most accurate, efficient, and reliable model for deployment.


