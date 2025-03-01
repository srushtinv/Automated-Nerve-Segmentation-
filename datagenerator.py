import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Define paths
input_dir = 'data/'  # Directory containing all your images
affected_dir = 'data/affected'  # Folder to store affected images
not_affected_dir = 'data/not_affected'  # Folder to store not affected images

# Create the directories for affected and not affected (if they don't exist)
os.makedirs(affected_dir, exist_ok=True)
os.makedirs(not_affected_dir, exist_ok=True)

# Load a pre-trained model (e.g., MobileNetV2)
model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=True)

# Function to preprocess image for prediction
def preprocess_image(img_path):
    # Load the image and resize it to (224, 224) which is the expected input size for MobileNetV2
    img = image.load_img(img_path, target_size=(224, 224))
    # Convert the image to a numpy array and add an extra dimension to match the model's input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess the image as required by MobileNetV2
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Classify and move images
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)

    # Ensure the file is a valid image and is not a directory
    if not os.path.isdir(img_path) and img_name.lower().endswith('.jpg'):
        try:
            # Preprocess the image
            img_array = preprocess_image(img_path)

            # Predict the class using the pre-trained model
            preds = model.predict(img_array)
            decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]
            label = decoded_preds[1].lower()  # Get the predicted label

            # Add a more general classification logic for detecting damage
            # These are just example labels, and you can adjust as needed
            damage_keywords = ['damaged', 'broken', 'cracked', 'affected', 'defective', 'injured']

            # Check if any damage-related keyword appears in the predicted label
            if any(keyword in label for keyword in damage_keywords):
                destination = affected_dir
            else:
                destination = not_affected_dir

            # Move the image to the appropriate folder based on the classification result
            shutil.move(img_path, os.path.join(destination, img_name))
            print(f"Moved {img_name} to {destination}")

        except Exception as e:
            print(f"Error processing image {img_name}: {e}")
