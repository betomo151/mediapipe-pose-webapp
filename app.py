import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
from moviepy.editor import VideoFileClip

# Mediapipe のポーズ推定の設定
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Streamlit アプリのタイトル
st.title("動画内の身体特徴点推定")

# 動画ファイルのアップロード
uploaded_video = st.file_uploader("動画をアップロード", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # 一時ファイルとして保存
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_video.read())
        temp_file_path = temp_file.name

    # OpenCV で動画を読み込み
    cap = cv2.VideoCapture(temp_file_path)
    if not cap.isOpened():
        st.error("動画の読み込みに失敗しました。")
    else:
        # 動画情報の取得
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 書き出し先の動画ファイルパス
        output_video_path = "/tmp/output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # BGRからRGBに変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ポーズ推定
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                # 姿勢のランドマークを描画
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 結果を動画ファイルに書き込む
            out.write(frame)

        # 動画処理完了
        cap.release()
        out.release()

        # 動画再生
        st.video(output_video_path)

        # 一時ファイル削除
        os.remove(temp_file_path)
