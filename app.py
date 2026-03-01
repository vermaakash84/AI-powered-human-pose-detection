import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import os
import numpy as np
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(page_title="Pose Detection App", layout="centered")

st.title("🧍 Human Pose Detection Web App")
st.write("Upload a video to detect full-body pose using MediaPipe FULL model.")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov"])


# =========================
# Load Pose Model (Cached)
# =========================
@st.cache_resource
def load_landmarker():
    model_path = "pose_landmarker_full.task"

    if not os.path.exists(model_path):
        st.info("Downloading pose model... Please wait...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
            model_path
        )
        st.success("Model downloaded successfully!")

    base_options = python.BaseOptions(
        model_asset_path=model_path
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    return vision.PoseLandmarker.create_from_options(options)


if uploaded_file is not None:

    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    input_video_path = tfile.name
    output_video_path = "output_pose.mp4"

    st.info("Processing video... Please wait.")

    landmarker = load_landmarker()

    # =========================
    # Pose Connections
    # =========================
    POSE_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,7),
        (0,4),(4,5),(5,6),(6,8),
        (9,10),
        (11,12),
        (11,13),(13,15),
        (12,14),(14,16),
        (15,17),(16,18),
        (11,23),(12,24),
        (23,24),
        (23,25),(25,27),
        (24,26),(26,28),
        (27,29),(29,31),
        (28,30),(30,32)
    ]

    cap = cv2.VideoCapture(input_video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # FPS fallback protection
    if fps is None or fps == 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # ✅ FIXED TIMESTAMP LOGIC
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Monotonically increasing timestamp (REQUIRED by MediaPipe VIDEO mode)
        timestamp_ms = int((frame_count / fps) * 1000)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            for pose_landmarks in result.pose_landmarks:

                landmark_points = []
                for landmark in pose_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    landmark_points.append((x, y))

                # Draw skeleton
                for start_idx, end_idx in POSE_CONNECTIONS:
                    if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                        cv2.line(
                            frame,
                            landmark_points[start_idx],
                            landmark_points[end_idx],
                            (0, 255, 0),
                            3
                        )

                # Draw landmarks
                for point in landmark_points:
                    cv2.circle(frame, point, 5, (0, 0, 255), -1)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    st.success("Processing Complete!")

    st.video(output_video_path)

    with open(output_video_path, "rb") as file:
        st.download_button(
            label="Download Processed Video",
            data=file,
            file_name="pose_output.mp4",
            mime="video/mp4"
        )
