# 🫁 Pneumonia Detection Using Chest X-Ray Images

## 📌 Overview

This project implements an **AI-based medical image classification system** to detect **pneumonia from chest X-ray images**. It uses **Convolutional Neural Networks (CNN)** and **transfer learning with MobileNetV2** to achieve high accuracy. The trained model is deployed as a **web application using Streamlit**, enabling real-time predictions through a simple user interface.

---

## 🎯 Problem Statement

Pneumonia is a serious respiratory infection that requires early and accurate diagnosis. Manual analysis of chest X-ray images is time-consuming and depends heavily on expert radiologists. This project aims to assist diagnosis by automatically classifying chest X-ray images as **Normal** or **Pneumonia** using deep learning.

---

## 🚀 Features

* Automated pneumonia detection from chest X-ray images
* Binary classification: **Normal / Pneumonia**
* CNN + MobileNetV2 transfer learning
* High accuracy with optimized model size
* Interactive Streamlit web interface
* Real-time image upload and prediction

---

## 🧠 Technologies Used

| Category             | Tools / Technologies |
| -------------------- | -------------------- |
| Programming Language | Python               |
| Deep Learning        | TensorFlow, Keras    |
| Model Architecture   | CNN, MobileNetV2     |
| Image Processing     | OpenCV, PIL          |
| Evaluation           | Scikit-learn         |
| Visualization        | Matplotlib, Seaborn  |
| Web Deployment       | Streamlit            |
| Version Control      | Git, GitHub          |

---

## 📂 Project Structure

```
Disease_Prediction_Project/
│
├── app.py                  # Streamlit web app
├── train_model.py          # Model training script
├── evaluate.py             # Model evaluation script
├── requirements.txt        # Dependencies
│
├── model/
│   └── disease_model.h5    # Trained model
│
├── dataset/
│   └── train/
│       ├── normal/
│       └── pneumonia/
│
└── README.md
```

---

## 📊 Dataset

* **Source:** Kaggle – Chest X-Ray Pneumonia Dataset
* **Classes:**

  * Normal
  * Pneumonia
* Images are organized into class-wise folders for supervised learning.

---

## ⚙️ Model Architecture

* MobileNetV2 pretrained on ImageNet
* Global Average Pooling
* Fully connected dense layer
* Sigmoid activation for binary classification
* Optimized by freezing most base layers to reduce model size and training time

---

## 🏋️ Model Training

* Image size: 224 × 224
* Batch size: 32
* Optimizer: Adam
* Loss function: Binary Cross-Entropy
* Epochs: 10
* Model saved without optimizer state to reduce size

---

## 📈 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

These metrics ensure reliable medical classification performance.

---

## 🌐 Web Application (Streamlit)

The Streamlit app allows users to:

1. Upload a chest X-ray image
2. Automatically preprocess the image
3. Get real-time prediction with confidence score

---

## ▶️ How to Run Locally

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train the model (run once)

```bash
python train_model.py
```

### 3️⃣ Run the web app

```bash
python -m streamlit run app.py
```

Open browser at:

```
http://localhost:8501
```

---

## ☁️ Deployment

The project is deployed using **Streamlit Community Cloud**, enabling live access via a public URL.
The trained model is loaded directly in the cloud for real-time inference.

---

## ⚠️ Disclaimer

This application is intended **for educational and research purposes only** and should not be used as a substitute for professional medical diagnosis.

---

## 📌 Future Enhancements

* Grad-CAM heatmap for explainable AI
* Multi-class lung disease classification
* Integration with hospital management systems
* Mobile application support

---

## 👨‍💻 Author

**Aditya Raj**
Final Year Engineering Student
ECE | AI & Software Development Enthusiast

---

## ⭐ Acknowledgements

* Kaggle for the dataset
* TensorFlow & Keras documentation
* Streamlit Community

---

© 2026 Aditya Raj. All rights reserved.
