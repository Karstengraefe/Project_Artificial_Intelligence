# Importing the model and libaries
import os
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the desired dimensions for the resized images
img_width, img_height = 128, 128

# Load and preprocess the single input image
def preprocess_input_image(image_path):
    input_image = Image.open(image_path).convert('L')  # Convert to grayscale
    input_image = input_image.resize((img_width, img_height))
    input_image = np.array(input_image) / 255.0  # Normalize the pixel values
    input_image = np.expand_dims(input_image, axis=0)
    input_image = np.expand_dims(input_image, axis=-1)  # Add a single channel dimension
    return input_image

# Load the saved model
model = tf.keras.models.load_model('emotion_detection_model.keras')

# Prompt the user to enter the file path of the input image
input_path = input("Enter the file path of the input image: ")

# Preprocess the input image
input_image = preprocess_input_image(input_path)

# Make the prediction
emotion_labels = ['happy', 'sad', 'surprised']  # List of emotion labels
prediction = model.predict(input_image)
predicted_emotion = emotion_labels[np.argmax(prediction)]

# Print the predicted emotion
print("The predicted emotion is:", predicted_emotion)