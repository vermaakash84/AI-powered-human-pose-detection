import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import os
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(page_title="Pose Detection App", layout="centered")

st.title("🧍 Human Pose Detection Web App")
st.write("Upload a video to detect full-body pose using MediaPipe FULL model.")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov"])

if uploaded_file is not None:

    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    input_video_path = tfile.name
    output_video_path = "output_pose.mp4"

    st.info("Processing video... Please wait.")

    # =========================
    # Load Pose Model
    # =========================
    base_options = python.BaseOptions(
        model_asset_path="pose_landmarker_full.task"
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    landmarker = vision.PoseLandmarker.create_from_options(options)

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
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_timestamp = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        result = landmarker.detect_for_video(mp_image, frame_timestamp)

        if result.pose_landmarks:
            for pose_landmarks in result.pose_landmarks:

                landmark_points = []
                for landmark in pose_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    landmark_points.append((x, y))

                # Draw skeleton
                for connection in POSE_CONNECTIONS:
                    start_idx, end_idx = connection
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
        frame_timestamp += int(1000 / fps)

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
