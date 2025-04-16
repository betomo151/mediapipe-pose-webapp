import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# MediapipeのPoseモジュールをセットアップ
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Streamlitにアップロードされた動画を表示
st.title('Mediapipe Pose Estimation WebApp')

video_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
if video_file is not None:
    # アップロードされた動画をOpenCVで読み込み
    video_bytes = video_file.read()
    nparr = np.frombuffer(video_bytes, np.uint8)
    video = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 動画のフレームごとにポーズ推定
    frame_rgb = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # ポーズの描画
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 結果の表示
    st.image(frame_rgb)
