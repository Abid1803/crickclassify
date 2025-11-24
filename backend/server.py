import os
import json
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

app = Flask(__name__)
CORS(app)

MODEL_PATH = "cricknet_model.keras"
DICT_PATH = "class_dictionary.json"

model = load_model(MODEL_PATH)

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
    face_crop = rgb[y:y+h, x:x+w]
    face_crop = cv2.resize(face_crop, IMG_SIZE)
    face_crop = face_crop / 255.0
    face_crop = np.expand_dims(face_crop, axis=0)
    return face_crop

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API Running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    processed = preprocess_image(img)
    if processed is None:
        return jsonify({"error": "Face detection failed"}), 400

    prediction = model.predict(processed)
    idx = int(np.argmax(prediction))
    return jsonify({
        "predicted_class": index_to_class[idx],
        "confidence": float(prediction[0][idx])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
