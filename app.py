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
    # アップロードされた動画ファイルをOpenCVで読み込む
    video_bytes = video_file.read()
    nparr = np.frombuffer(video_bytes, np.uint8)
    
    # 動画をOpenCVでデコード
    video = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # ビデオキャプチャオブジェクトを作成
    cap = cv2.VideoCapture(video_file)

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
