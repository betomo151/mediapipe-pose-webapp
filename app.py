import streamlit as st
import tempfile
import cv2
import numpy as np
from io import BytesIO

def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 書き出し先：Memory buffer (一時ファイルにしてから読むのが確実)
    out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out = cv2.VideoWriter(out_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 例：MediaPipeの処理がここに入る
        out.write(frame)

    cap.release()
    out.release()

    # 再生できるように読み込み直す
    video_bytes = open(out_file.name, 'rb').read()
    return video_bytes

st.title("MediaPipe Pose 動画処理 WebApp")

uploaded_file = st.file_uploader("動画をアップロードしてください", type=["mp4", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.info("ポーズ検出処理中...")
    result_video = process_video(uploaded_file)
    st.success("✅ポーズ検出完了！以下の動画で確認できます。")
    st.video(result_video)
