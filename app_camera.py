import streamlit as st
import cv2
from services.pipeline import process_image

st.set_page_config(page_title="Live Camera", layout="wide")

st.title("📷 Live Camera Detection")

start = st.button("Start Camera")

if start:

    cap = cv2.VideoCapture(0)

    frame_placeholder = st.empty()

    stop_button = st.button("Stop Camera")

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        result = process_image(frame)

        frame_placeholder.image(
            result["image"],
            channels="BGR",
            use_container_width=True
        )

        if stop_button:
            break

    cap.release()