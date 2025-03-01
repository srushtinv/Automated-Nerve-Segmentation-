from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('nerve_ultrasound_classifier.h5')

# Create a Flask app
app = Flask(__name__)

# Define the allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if the uploaded file is an image
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to serve the welcome page on `/`
@app.route('/')
def welcome():
    return render_template('welcome.html')  # Ensure you have a 'welcome.html' template

# Route to serve the homepage (index.html) on `/home`
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # If no file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # If the file is allowed, process it
    if file and allowed_file(file.filename):
        # Save the image temporarily
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)
        
        # Preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))  # Resize image
        img_array = image.img_to_array(img) / 255.0  # Convert to array and normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Predict with the model
        prediction = model.predict(img_array)
        
        # Return prediction result
        result = 'Affected' if prediction[0] > 0.5 else 'Not Affected'
        return jsonify({'result': result})
    else:
        return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)
