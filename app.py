import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# MediapipeのPoseモジュールをセットアップ
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Streamlitにアップロードされた動画を表示
st.title('Mediapipe Pose Estimation WebApp')

video_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
if video_file is not None:
    # 一時的なファイルを作成して動画を保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        temp_video_file.write(video_file.read())
        temp_video_path = temp_video_file.name

    # cv2.VideoCaptureで動画を開く
    cap = cv2.VideoCapture(temp_video_path)

    # 動画のフレームごとに処理
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mediapipeでポーズ推定
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # ポーズの描画
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 結果の表示
        st.image(frame_rgb)

    cap.release()
