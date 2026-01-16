# 🫁 Pneumonia Detection Using Chest X-Ray Images

## 📌 Overview

This project implements an **AI-based medical image classification system** to detect **pneumonia from chest X-ray images**. It uses **Convolutional Neural Networks (CNN)** and **transfer learning with MobileNetV2** to perform binary classification. The trained model is deployed as a **Streamlit web application**, enabling real-time predictions through a simple user interface.

---

## 🌍 Live Demo

🚀 **Try the application here:**  
👉 https://pneumonia-detection-using-chest-x-ray-images-yfgelhhxwk2ptlgfl.streamlit.app/

> Note: The app may take a few seconds to wake up if it is idle.

---

## 🎯 Problem Statement

Pneumonia is a serious respiratory infection that requires early and accurate diagnosis. Manual analysis of chest X-ray images is time-consuming and depends heavily on expert radiologists. This project aims to assist diagnosis by automatically classifying chest X-ray images as **NORMAL** or **PNEUMONIA** using deep learning techniques.

---

## 🚀 Features

* Automated pneumonia detection from chest X-ray images
* Binary classification: **NORMAL / PNEUMONIA**
* CNN with MobileNetV2 transfer learning
* Optimized model size for deployment
* Interactive Streamlit-based web interface
* Real-time image upload and prediction

---

## 🧠 Technologies Used

| Category             | Tools / Technologies |
| -------------------- | -------------------- |
| Programming Language | Python               |
| Deep Learning        | TensorFlow, Keras    |
| Model Architecture   | CNN, MobileNetV2     |
| Image Processing     | PIL, OpenCV          |
| Evaluation           | Scikit-learn         |
| Visualization        | Matplotlib           |
| Web Deployment       | Streamlit            |
| Version Control      | Git, GitHub          |

---

## 📂 Project Structure

```
Pneumonina-detection/
│
├── app.py                  # Streamlit web app
├── train_model.py          # Model training script
├── evaluate.py             # Model evaluation script
├── requirements.txt        # Python dependencies
│
├── model/
│   └── disease_model.h5    # Trained model
│
├── dataset/
│   └── train/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
└── README.md
```

✔ Folder names match actual class labels used during training.

---

## 📊 Dataset

* **Source:** Kaggle – Chest X-Ray Pneumonia Dataset
* **Classes:**

  * NORMAL
  * PNEUMONIA
* Images are organized into class-wise directories for supervised learning.

---

## ⚙️ Model Architecture

* MobileNetV2 pretrained on ImageNet
* Global Average Pooling layer
* Fully connected dense layer
* Sigmoid activation for binary classification
* Most base layers frozen to reduce overfitting and model size

---

## 🏋️ Model Training

* Image size: **224 × 224**
* Batch size: **32**
* Optimizer: **Adam**
* Loss function: **Binary Cross-Entropy**
* Epochs: **10**
* Class weights applied to handle dataset imbalance
* Model saved **without optimizer state** for deployment efficiency

---

## 📈 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Evaluation is performed on a validation split derived from the training dataset.

---

## 🌐 Web Application (Streamlit)

The Streamlit application allows users to:

1. Upload a chest X-ray image
2. Automatically preprocess the image
3. Receive a prediction with confidence score

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

### 3️⃣ Evaluate the model

```bash
python evaluate.py
```

### 4️⃣ Run the web app

```bash
python -m streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## ☁️ Deployment

The application is compatible with **Streamlit Community Cloud**.
Once deployed, users can access the model through a public URL for real-time inference.

---

## ⚠️ Disclaimer

This application is intended **for educational and research purposes only**.
Predictions may vary for X-ray images from external sources due to **domain shift** and should not be used for clinical diagnosis.

---

## 📌 Future Enhancements

* Grad-CAM heatmaps for model explainability
* Multi-class lung disease classification
* Training on multiple medical datasets to reduce domain shift
* Mobile application integration

---

## 👨‍💻 Author

**Aditya Raj**
ECE | AI & Software Development Enthusiast

---

## ⭐ Acknowledgements

* Kaggle for the dataset
* TensorFlow & Keras documentation
* Streamlit Community

---

© 2026 Aditya Raj. All rights reserved.
