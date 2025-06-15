import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = tf.keras.models.load_model('plant_disease_cnn_model.h5')

# Image size (must match the size used during training)
img_height, img_width = 224, 224

# Function to load and preprocess the image
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))  # Load image with target size
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch size
    img_array /= 255.0  # Rescale the pixel values (same as during training)
    return img_array

# Function to make a prediction
def predict_image(img_path):
    # Preprocess the image
    img_array = prepare_image(img_path)
    
    # Predict the class (returns probabilities)
    prediction = model.predict(img_array)
    
    # Interpret the result
    if prediction[0][0] < 0.5:
        print(f'The plant in {os.path.basename(img_path)} is Diseased')
    else:
        print(f'The plant in {os.path.basename(img_path)} is Healthy')

# Example usage: replace 'path_to_image' with the actual image path
img_path = '../test/Peach___Bacterial_spot/03c471c6-2d00-4315-ba13-a547e125866a___Rut._Bact.S 0857.JPG'
predict_image(img_path)
