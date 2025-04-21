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

    # 出力する動画のファイル名とコーデックを設定
    output_video_path = "/tmp/output_video.mp4"  # 一時ディレクトリに保存
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4形式で書き出し
    fps = cap.get(cv2.CAP_PROP_FPS)  # 元の動画のFPSを取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)  # 幅を半分に
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)  # 高さを半分に
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

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
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 処理したフレームを動画に書き出し
        out.write(frame)

    cap.release()
    out.release()

    # 出力した動画のパスを表示
    st.write(f"Video saved at: {output_video_path}")

    # 出力した動画を表示
    st.video(output_video_path)
