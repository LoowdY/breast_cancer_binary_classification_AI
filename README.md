# Breast Cancer Binary Classification Neural Network

This repository contains a machine learning project aimed at predicting breast cancer using a neural network model. The model performs binary classification to distinguish between malignant and benign tumors based on features extracted from digitized images of a fine needle aspirate (FNA) of a breast mass.

## Project Overview

The primary goal of this project is to develop a highly accurate predictive model that can assist in the early detection of breast cancer. The model is built using a sequential neural network architecture designed with Keras and TensorFlow backend.

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset, which is publicly available in the UCI Machine Learning Repository. It includes measurements from 569 instances, each with 30 numeric, predictive attributes and the binary class label indicating the diagnosis (malignant or benign).

## Model Architecture

The neural network model is constructed using the Sequential model from Keras. It consists of the following layers:

- Input layer with 30 nodes corresponding to the 30 features of the dataset.
- Two hidden layers, each with 8 nodes and ReLU activation functions.
- Dropout layers after each hidden layer with a dropout rate of 0.2 to prevent overfitting.
- Output layer with a single node with a sigmoid activation function to produce a binary output.

The model uses the Adam optimizer and binary crossentropy as the loss function. The performance metric used to evaluate the model is binary accuracy.


