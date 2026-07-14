import streamlit as st
import cv2
from ultralytics import YOLO

st.title("Real-Time Object Detection App")

# Load model
model = YOLO("D:/Computer_Vision_Proj/yolov5/runs/detect/train-26/weights/best.pt")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Camera not working")
        break

    results = model(frame)
    frame = results[0].plot()

    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()
