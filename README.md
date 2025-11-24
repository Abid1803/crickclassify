CrickClassify 🏏 — Cricketer Identification Using Deep Learning

Live Demo:
👉 https://crickclassify.netlify.app/

📌 Overview

CrickClassify is a lightweight deep-learning application that identifies Indian cricketers from an uploaded image.
It combines face detection, image preprocessing, and a convolutional neural network to output the player name along with a confidence score.

The goal of this project is to demonstrate a simple yet functional end-to-end ML deployment pipeline — from dataset cleaning to model serving and frontend integration.

🧠 Machine Learning Pipeline
1️⃣ Dataset Preparation

Training images were organized by player name

Each image was cleaned using:

MTCNN → detects exactly one face

Blurriness filtering using Laplacian variance

Automatic cropping and resizing to 224×224

2️⃣ Model Architecture

A lightweight CNN built on MobileNetV2 (pretrained on ImageNet):

MobileNetV2 (frozen base)

GlobalAveragePooling

Dense(256, ReLU)

Dropout(0.3)

Dense(num_classes, softmax)

This architecture provides fast inference and small model size, ideal for web deployment.

3️⃣ Training Info

Optimizer: Adam

Loss: Categorical Crossentropy

Epochs: 15

Batch size: 32

Steps per epoch: 15 (adjusted for dataset size)

4️⃣ Inference Workflow

When a user uploads an image:

MTCNN detects the face

The face is cropped & normalized

TensorFlow model returns a probability vector

The highest-confidence class is chosen

The result is shown in the UI

🌐 Deployment Architecture
Frontend — Netlify

Pure HTML + JS interface

Minimalistic JetBrains Mono theme

Sends the uploaded image to backend API

Backend — Render

Flask API serves predictions

TensorFlow loads the .keras model

MTCNN + OpenCV perform preprocessing

Returns JSON response with:

predicted_class

confidence

📁 Project Structure
crickclassify/
│
├── backend/
│   ├── server.py
│   ├── cricknet_model.keras
│   ├── class_dictionary.json
│   ├── requirements.txt
│
└── frontend/
    └── index.html

👤 Author

Mohammad Abid (maxEpoch)
Portfolio: https://maxepoch.netlify.app/

🚀 Live Project Link

👉 https://crickclassify.netlify.app/
