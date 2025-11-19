import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ----------------------------------------------------
# PAGE CONFIG 
# ----------------------------------------------------
st.set_page_config(page_title="Smart Waste Classifier", layout="centered")

custom_css = """
<style>

html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

/* Remove Streamlit default top padding */
.main > div {
    padding-top: 0rem !important;
}

/* Remove file uploader WHITE TEXT BAR */
.css-9oya0r, .e1y61itm0, .stFileUploader label div:nth-child(2) {
    display: none !important;
}

/* Center title */
.title-style {
    text-align: center;
    font-size: 40px;
    color: #32CD32;
    font-weight: 700;
    margin-top: 10px;
}

/* Subtitle */
.subtitle-style {
    text-align: center;
    font-size: 18px;
    color: #cccccc;
    margin-bottom: 25px;
}

/* Result box */
.result-box {
    padding: 18px;
    border-radius: 12px;
    font-size: 22px;
    font-weight: 600;
    text-align: center;
}

/* Success (Recyclable) */
.recyclable {
    background-color: rgba(0, 255, 150, 0.15);
    color: #00ff9a;
    border: 1px solid #00ff9a;
}

/* Organic (Error) */
.organic {
    background-color: rgba(255, 80, 80, 0.15);
    color: #ff6b6b;
    border: 1px solid #ff6b6b;
}

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------
@st.cache_resource
def load_waste_model():
    return tf.keras.models.load_model("waste_cnn_model.h5")

model = load_waste_model()

# ----------------------------------------------------
# HEADER
# ----------------------------------------------------
st.markdown("<h1 class='title-style'>♻ Smart Waste Classification App</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-style'>Upload your waste image this app classifies waste as **Organic** or **Recyclable** using a CNN model.</p>", unsafe_allow_html=True)

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
st.sidebar.header("📈 Model Information")

try:
    st.sidebar.image("accuracy_plot.png", caption="Training Accuracy", use_container_width=True)
except:
    st.sidebar.info("Upload accuracy_plot.png to display training graph.")

# ----------------------------------------------------
# FILE UPLOADER
# ----------------------------------------------------
uploaded_file = st.file_uploader("Upload Waste Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=280)

    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((128, 128))

    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    with st.spinner("Classifying..."):
        prediction = float(model.predict(img_batch)[0][0])

    st.subheader("Result:")

    if prediction > 0.5:
        result_html = "<div class='result-box recyclable'>♻ Recyclable Waste</div>"
    else:
        result_html = "<div class='result-box organic'>🍃 Organic Waste</div>"

    st.markdown(result_html, unsafe_allow_html=True)

    st.write(f"**Model Confidence:** `{prediction:.4f}`")

else:
    st.info("Please upload an image to begin.")
