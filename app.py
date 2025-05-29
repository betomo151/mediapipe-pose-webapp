import streamlit as st
import tempfile
import cv2
import numpy as np
from io import BytesIO

def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.flush()

    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("動画の読み込みに失敗しました。")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25  # fallback

    # より互換性の高いコーデック（環境による）
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    if frame_count == 0:
        st.error("フレームが書き込まれませんでした。動画が空か、読み込みエラーの可能性があります。")
        return None

    with open(out_file.name, "rb") as f:
        return f.read()

