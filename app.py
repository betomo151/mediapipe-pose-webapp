import streamlit as st
import tempfile
import cv2
from video_utils import process_video

st.title("🧍 Mediapipe Pose Detection Web App")

video_file = st.file_uploader("動画をアップロードしてください", type=["mp4", "mov", "avi"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    st.info("処理中...しばらくお待ちください")

    frames = process_video(tfile.name)

    st.success("処理完了！")

    height, width, _ = frames[0].shape
    out_path = tfile.name + "_out.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()

    st.video(out_path)
