import os

import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

MODEL_PATH = os.path.join("model", "disease_model.h5")
model = load_model(MODEL_PATH)

st.title("Pneumonia Detection Using Chest X-ray")

file = st.file_uploader("Upload Chest X-ray", type=["jpg","png"])

if file:
    img = Image.open(file).resize((224,224))
    img = np.array(img)/255.0
    img = img.reshape(1,224,224,3)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        st.error(f"Pneumonia Detected ({prediction*100:.2f}%)")
    else:
        st.success(f"Normal ({(1-prediction)*100:.2f}%)")
