import streamlit as st
import mediapipe as mp
import cv2
import tempfile
import numpy as np
import os

st.set_page_config(page_title="Mediapipe Pose WebApp", layout="centered")
st.title("📹 Mediapipe Pose WebApp")

video_file = st.file_uploader("🎞️ 動画をアップロードしてください", type=["mp4", "mov", "avi"])

if video_file is not None:
    # 一時ファイルとして保存
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())

    # 動画読み込み
    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("❌ 動画を開けませんでした。形式やファイルを確認してください。")
    else:
        st.info("✅ 動画を読み込みました！Pose推定を開始します...")

        # Mediapipe初期化
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        # 出力ファイル準備
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = os.path.join(tempfile.gettempdir(), 'processed_video.mp4')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 色変換と推定
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                # 骨格描画
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # 書き込み
                out.write(frame)

        cap.release()
        out.release()

        st.success("✅ 処理が完了しました！下の動画で確認できます：")
        st.video(output_path)
