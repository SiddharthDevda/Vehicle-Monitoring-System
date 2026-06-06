import streamlit as st
import cv2
import tempfile
from services.pipeline import process_image

st.set_page_config(page_title="Video Detection", layout="wide")

st.title("🎥 Video Vehicle Detection")

uploaded_video = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_video:

    temp_file = tempfile.NamedTemporaryFile(delete=False)

    temp_file.write(uploaded_video.read())

    cap = cv2.VideoCapture(temp_file.name)

    frame_placeholder = st.empty()

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

    cap.release()

    st.success("Video Processing Completed")