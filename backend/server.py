import os
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify, render_template
from mtcnn import MTCNN
import json
import io
from PIL import Image

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "cricknet_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Load class dictionary
CLASS_DICT_PATH = "class_dictionary.json"
with open(CLASS_DICT_PATH, "r") as f:
    class_dict = json.load(f)

# Initialize MTCNN detector
detector = MTCNN()

def prepare_image(image_bytes):
    """Preprocesses the image to match training conditions exactly."""
    try:
        # Convert bytes to PIL Image, then to numpy array (RGB)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = np.array(image)

        # Detect faces
        faces = detector.detect_faces(image)
        if not faces:
            return None # No face detected

        # Assume the first detected face is the target
        x, y, w, h = faces[0]['box']
        # Ensure coordinates are within bounds
        x, y = max(0, x), max(0, y)
        face_img = image[y:y+h, x:x+w]

        # Resize to model's expected input size
        face_img = cv2.resize(face_img, (224, 224))

        img_array = tf.keras.preprocessing.image.img_to_array(face_img)

        # --- CRITICAL FIX HERE ---
        # Training used simple / 255.0 scaling. Server must do the same.
        # DO NOT use mobilenet_v2.preprocess_input here.
        img_array = img_array / 255.0
        # -------------------------

        # Expand dims to create a batch of size 1: (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image_bytes = file.read()
        processed_image = prepare_image(image_bytes)

        if processed_image is None:
             # Handle case where MTCNN couldn't find a face
             return jsonify({'player': 'Could not detect a face in the image', 'confidence': 0.0})

        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        # Threshold confidence
        if confidence < 0.7:
            player_name = "Unidentified Player"
        else:
            # Reverse class dict to get name from index
            index_to_class = {v: k for k, v in class_dict.items()}
            player_name = index_to_class.get(predicted_class_index, "Unknown")
            # Format name nicely (e.g., "virat_kohli" -> "Virat Kohli")
            player_name = player_name.replace('_', ' ').title()

        return jsonify({'player': player_name, 'confidence': f"{confidence:.2f}"})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

# Remove app.run() for production. Gunicorn will handle running the app.
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=10000)
