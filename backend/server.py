import os
import json
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from mtcnn import MTCNN

app = Flask(__name__)
CORS(app)

# Absolute paths (important for Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cricknet_model.keras")
DICT_PATH = os.path.join(BASE_DIR, "class_dictionary.json")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class dictionary
with open(DICT_PATH, "r") as f:
    class_dict = json.load(f)

index_to_class = {v: k for k, v in class_dict.items()}

detector = MTCNN()
IMG_SIZE = (224, 224)

def preprocess_image(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    if len(faces) != 1:
        return None

    x, y, w, h = faces[0]["box"]
    face = rgb[y:y+h, x:x+w]

    face = cv2.resize(face, IMG_SIZE)
    face = face / 255.0
    face = np.expand_dims(face, axis=0)
    return face


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API Running"})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:   # IMPORTANT FIX
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    processed = preprocess_image(img)

    if processed is None:
        return jsonify({"error": "Face not detected or multiple faces found"}), 400

    prediction = model.predict(processed)
    idx = int(np.argmax(prediction))
    confidence = float(prediction[0][idx])

    return jsonify({
        "predicted_class": index_to_class[idx],
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
