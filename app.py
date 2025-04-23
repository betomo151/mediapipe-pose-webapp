import streamlit as st
import mediapipe as mp
import cv2
import tempfile
import numpy as np

st.title("Mediapipe Pose WebApp")

video_file = st.file_uploader("動画をアップロードしてください", type=["mp4", "mov", "avi"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("動画を開けませんでした。形式やファイルを確認してください。")
    else:
        st.success("動画を読み込みました！")

        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        stframe = st.empty()

        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                stframe.image(frame, channels="BGR")

        cap.release()
