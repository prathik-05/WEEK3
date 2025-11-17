import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("♻️ Smart Waste Classification – Week 3 WebApp")

model = tf.keras.models.load_model("waste_cnn_model.h5")
class_names = ['Organic Waste', 'Recyclable Waste']

uploaded = st.file_uploader("Upload Waste Image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).resize((128,128))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_batch)
    idx = np.argmax(prediction[0])
    confidence = prediction[0][idx] * 100

    st.success(f"Prediction: **{class_names[idx]}** ({confidence:.2f}% confidence)")
