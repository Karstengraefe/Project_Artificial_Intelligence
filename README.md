# Project_Artificial_Intelligence

Emotion Detection Model

This repository contains the test and train dataset, model and a script for an Emotion Detection system using Python and TensorFlow. The model is trained on a dataset of facial images categorized into three different emotions (happy, sad, suprised).
Features

    Train an emotion detection model using Convolutional Neural Networks (CNN) architecture.
    Evaluate the model's performance on a test dataset.
    Make predictions on single input images to detect emotions.

Prerequisites

Before running the code, ensure you have the following libraries installed:

    TensorFlow
    NumPy
    Pillow
    Matplotlib
    scikit-learn

You can install the required libraries by running the following command:
python

pip install tensorflow numpy pillow matplotlib scikit-learn

Dataset

The dataset is included in the repository and is located in the dataset directory. It contains facial images labeled with different emotions. The dataset is already organized into two directories:

    Train Directory: This directory contains subdirectories for each emotion category. Each subdirectory contains the corresponding facial images for that emotion.

    Test Directory: This directory has the same structure as the train directory. It contains subdirectories for each emotion and the corresponding test images.

Training the Model

To train the emotion detection model, follow these steps:

    Open the model03.ipynb file in Jupyter Notebook or any other Python IDE.

    Modify the train_dir and test_dir variables to point to the provided dataset directories.

    Run the code cells in the notebook to train the model, evaluate its performance, and save the trained model.

Making Predictions

To make predictions on single input images, follow these steps:

    Ensure you have the trained model file (emotion_detection_model.keras) in the same directory as the emotion_prediction.py script.

    Run the emotion_prediction.py script.

    When prompted, enter the file path of the input image you want to predict the emotion for.

    The script will preprocess the input image, make the prediction using the trained model, and display the predicted emotion.

Trained Model

The trained model file (emotion_detection_model.keras) is included in the repository.
Model Architecture

The emotion detection model is built using the Sequential API from TensorFlow. It consists of several convolutional and pooling layers followed by dense layers for classification. The model architecture is described in the model03.ipynb notebook.
Results

The trained model achieves a test accuracy of 82.89% and a test loss of 0.6384. The precision, recall, and F1-score for the emotion detection task are also calculated and printed.
License

This project is licensed under the GNU General Public License v3.0. 