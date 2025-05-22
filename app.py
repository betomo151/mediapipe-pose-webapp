import streamlit as st
import mediapipe as mp
import cv2
import tempfile
import os
import numpy as np
from moviepy.editor import VideoFileClip

st.title("🎥 Mediapipe Pose WebApp")

video_file = st.file_uploader("動画をアップロードしてください（mp4推奨）", type=["mp4", "mov", "avi"])

if video_file is not None:
    # 一時的に動画を保存
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())

    input_path = tfile.name

    # OpenCVで動画読み込み
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("❌ 動画を開けませんでした。ファイル形式や内容を確認してください。")
    else:
        st.success("✅ 動画を読み込みました！処理中...")

        # 出力用動画ファイルの設定
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out_path = output_temp.name
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        # Mediapipe 初期化
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # BGR → RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                # 骨格描画
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                out.write(frame)

        cap.release()
        out.release()

        st.success("✅ 処理完了！以下で再生できます：")
        st.video(out_path)
