import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import tempfile

# Mediapipe設定
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# StreamlitによるUI
st.title("Mediapipe 身体特徴点推定 Webアプリ")
st.write("動画ファイルをアップロードして、身体特徴点を推定します。")

# アップロード機能
uploaded_video = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi"])

# アップロードされたファイルがある場合
if uploaded_video is not None:
    # 一時的なファイルとして保存
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    
    # 動画ファイルの読み込み
    video = VideoFileClip(video_path)

    # 変換用の動画処理関数
    def process_frame(frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        return frame

    # 動画フレーム処理
    processed_video = video.fl_image(process_frame)

    # 動画をStreamlitに表示
    st.video(processed_video.write_videofile(tempfile.NamedTemporaryFile(delete=False).name))
