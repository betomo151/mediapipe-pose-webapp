import streamlit as st
import mediapipe as mp
import tempfile
import moviepy.editor as mp_editor
import numpy as np
import cv2
import os

st.title("🎥 MediaPipe Pose 推定 Webアプリ")

uploaded_file = st.file_uploader("動画をアップロードしてください", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # 動画を一時ファイルに保存
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    st.video(tfile.name)
    st.info("ポーズを解析しています...")

    # 動画を読み込み
    video = mp_editor.VideoFileClip(tfile.name)
    fps = video.fps

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    frames = []
    with mp_pose.Pose(static_image_mode=False) as pose:
        for frame in video.iter_frames(fps=fps, dtype="uint8"):
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    st.success("完了！")

    # 書き出し（動画として）
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out_clip = mp_editor.ImageSequenceClip(frames, fps=fps)
    out_clip.write_videofile(out_path, codec="libx264")

    # 表示
    st.video(out_path)
