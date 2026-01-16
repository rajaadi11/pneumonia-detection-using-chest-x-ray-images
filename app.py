import os
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

MODEL_PATH = os.path.join("model", "disease_model.h5")
model = load_model(MODEL_PATH)

st.title("Pneumonia Detection Using Chest X-ray")
st.write("Upload a chest X-ray image to detect Pneumonia.")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        img = ImageOps.autocontrast(img)
        img = img.resize((224, 224))

        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)[0][0]

        THRESHOLD = 0.55

        st.subheader("Prediction Result")
        st.write(f"Model confidence: **{prediction * 100:.2f}%**")

        if prediction >= THRESHOLD:
            st.error("Pneumonia Detected")
        else:
            st.success("Normal")

        st.image(img, caption="Uploaded Chest X-ray", use_container_width=True)

    except Exception as e:
        st.error("An error occurred while processing the image.")
        st.exception(e)

st.caption(
    "Predictions may vary for X-ray images from external sources. "
    "This application is for educational and research purposes only."
)
