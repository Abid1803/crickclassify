from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
from mtcnn import MTCNN
import json

app = Flask(__name__)
CORS(app)

# Load model
model = tf.keras.models.load_model("cricknet_model.keras")

# Load class dictionary
with open("class_dictionary.json") as f:
    class_dict = json.load(f)

index_to_class = {v: k for k, v in class_dict.items()}

detector = MTCNN()
IMG_SIZE = (224, 224)

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(image)
    if len(faces) != 1:
        return None, "No face or multiple faces detected"

    x, y, w, h = faces[0]['box']
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, IMG_SIZE)
    face = face / 255.0
    face = np.expand_dims(face, axis=0)

    return face, None


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file found"}), 400

    file = request.files["file"]
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    processed, err = preprocess_image(img)
    if err:
        return jsonify({"error": err}), 400

    preds = model.predict(processed)
    class_index = np.argmax(preds[0])
    class_name = index_to_class[class_index]

    return jsonify({
        "prediction": class_name,
        "confidence": float(preds[0][class_index])
    })


@app.route("/", methods=["GET"])
def home():
    return "Cricketer Classifier API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
