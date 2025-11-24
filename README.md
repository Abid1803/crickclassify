# CrickClassify 🏏 — Indian Cricketer Identification Using Deep Learning

[![Live Demo](https://img.shields.io/badge/Live_Demo-Visit_Now-blue?style=for-the-badge&logo=netlify)](https://crickclassify.netlify.app/)

A lightweight, end-to-end deep learning web application that instantly recognizes popular **Indian cricketers** from any uploaded photo using face detection and a custom-trained CNN.

👉 **Live Demo**: https://crickclassify.netlify.app/

## 📌 Overview

CrickClassify combines modern computer vision techniques to deliver fast and accurate cricketer identification directly in the browser-to-server pipeline:

- Face detection → Image preprocessing → CNN inference → Player name + confidence score

Perfect for cricket fans, this project also serves as a clean, production-ready example of a full ML deployment pipeline — from raw data cleaning to live web serving.

## 🧠 Machine Learning Pipeline

### 1️⃣ Dataset Preparation
- Images organized by player name
- Strict cleaning using:
  - **MTCNN** – ensures exactly one face per image
  - Blurriness filtering (Laplacian variance threshold)
  - Automatic face cropping + resizing to **224×224**

### 2️⃣ Model Architecture (Lightweight & Fast)
Built on transfer learning with **MobileNetV2** (pretrained on ImageNet):

```text
MobileNetV2 (frozen base layers)
├── GlobalAveragePooling2D
├── Dense(256, activation='relu')
├── Dropout(0.3)
└── Dense(num_classes, activation='softmax')
```

### 3️⃣ Training Details

| Hyperparameter      | Value                          |
|---------------------|--------------------------------|
| Optimizer           | Adam                           |
| Loss                | Categorical Crossentropy       |
| Epochs              | 15                             |
| Batch Size          | 32                             |
| Steps per Epoch     | 15 (balanced for dataset size) |

### 4️⃣ Inference Workflow

1. User uploads an image  
2. MTCNN detects and extracts the face  
3. Face is cropped, resized, and normalized  
4. TensorFlow Keras model predicts probabilities  
5. Highest-confidence class + score displayed instantly  

### 🌐 Deployment Architecture

```text
Frontend (Netlify)                Backend (Render)
HTML + Vanilla JS   ←→   Flask API + TensorFlow
Minimal JetBrains Mono theme         Loads .keras model
Sends image as base64                MTCNN + OpenCV preprocessing
                                     Returns JSON {predicted_class, confidence}
```
- **Frontend**: Static site hosted on Netlify (zero server management)
- **Backend**: Flask API hosted on Render (free tier friendly)

### 📁 Project Structure

```text
crickclassify/
│
├── backend/
│   ├── server.py              # Flask API
│   ├── cricknet_model.keras   # Trained model
│   ├── class_dictionary.json  # Class name mapping
│   └── requirements.txt       # Python dependencies
│
└── frontend/
    └── index.html             # Single-page UI
```
### 👤 Author

**Mohammad Abid** (@maxEpoch)  
Portfolio: [https://maxepoch.netlify.app](https://maxepoch.netlify.app)
