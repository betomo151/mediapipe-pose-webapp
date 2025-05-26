import streamlit as st
import mediapipe as mp
import cv2
import tempfile
import os

st.title("MediaPipe Pose 動画処理 WebApp")

video_file = st.file_uploader("🎥 動画をアップロードしてください（.mp4/.mov）", type=["mp4", "mov"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    input_path = tfile.name

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("❌ 動画の読み込みに失敗しました。")
    else:
        st.info("⏳ ポーズ検出処理中...")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = output_file.name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # または 'avc1'
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # BGR → RGB で処理
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                # BGR フレームにそのまま描画
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # BGR のまま保存
                out.write(frame)

        cap.release()
        out.release()

        st.success("✅ ポーズ検出完了！以下の動画で確認できます。")
        st.video(output_path)
