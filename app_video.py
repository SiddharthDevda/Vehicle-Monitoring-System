import streamlit as st
import cv2
import tempfile
import os
from services.pipeline import process_image
print("APP_VIDEO_LOADED")

st.set_page_config(page_title="Video Detection", layout="wide")

st.title("🎥 Video Vehicle Detection")

uploaded_video = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_video:

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_video.read())
    temp_file.close()

    cap = cv2.VideoCapture(temp_file.name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    output_path = "processed_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    progress_bar = st.progress(0)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        result = process_image(frame, video_mode=True)

        out.write(result["image"])

        current_frame += 1

        # Terminal progress
        if current_frame % 30 == 0:
            print(f"Processed {current_frame}/{total_frames} frames")

        if total_frames > 0:
            progress_bar.progress(
                min(current_frame / total_frames, 1.0)
            )

    print("VIDEO LOOP FINISHED")

    cap.release()

    out.release()

    print("VIDEO SAVED")

    st.success("✅ Video Processing Completed")

    if os.path.exists(output_path):

        st.write(f"Saved file: {output_path}")

        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button(
                "📥 Download Processed Video",
                data=f,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

    else:
        st.error("❌ processed_video.mp4 was not created")