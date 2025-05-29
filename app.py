import streamlit as st
import tempfile
import cv2
import numpy as np

def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.flush()
    tfile.close()  # 重要

    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("動画の読み込みに失敗しました。")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not isinstance(fps, (int, float)) or fps <= 1:
        fps = 25  # fallback

    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'avc1'よりもmp4vの方が互換性あり
    out = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    if frame_count == 0:
        st.error("動画が空です。")
        return None

    with open(out_file.name, "rb") as f:
        return f.read()
