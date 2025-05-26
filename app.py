import streamlit as st
import tempfile
import cv2
import mediapipe as mp
import os
from moviepy.editor import VideoFileClip

st.title("🎥 Mediapipe Pose Processor")

video_file = st.file_uploader("動画をアップロードしてください（.mp4 or .mov）", type=["mp4", "mov"])

if video_file is not None:
    # 一時的に保存
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mov")
    input_temp.write(video_file.read())
    input_path = input_temp.name

    # .mov を .mp4 に変換
    mp4_path = input_path.replace(".mov", ".mp4")
    clip = VideoFileClip(input_path)
    clip.write_videofile(mp4_path, codec='libx264')  # H.264形式に強制変換

    # OpenCV で読み込み
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        st.error("❌ 動画を開けませんでした。")
    else:
        st.info("⏳ 骨格を抽出中...")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        with mp_pose.Pose(min_detection_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                out.write(frame)

        cap.release()
        out.release()

        st.success("✅ 完了！再生できます：")
        st.video(output_path)
