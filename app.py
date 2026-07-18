import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="0"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Object Detection App")

# Load model (repo se)
model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to numpy
    img_array = np.array(image)

    # Prediction
    results = model(img_array)

    # Plot result
    result_img = results[0].plot()

    st.image(result_img, caption="Detected Image", use_column_width=True)
