import streamlit as st
import cv2
import numpy as np
from video_utils import process_video
from io import BytesIO

st.title("🧍 Mediapipe Pose Detection Web App")

video_file = st.file_uploader("動画をアップロードしてください", type=["mp4", "mov", "avi"])

if video_file:
    st.info("処理中...しばらくお待ちください")

    # メモリ上に保存してOpenCVで読み込めるように
    video_bytes = np.frombuffer(video_file.read(), np.uint8)
    video_array = cv2.imdecode(video_bytes, cv2.IMREAD_COLOR)

    if video_array is None:
        st.error("動画を読み込めませんでした")
    else:
        # 一時ファイル作成せず、直接処理
        frames = process_video(video_file)

        st.success("処理完了！")

        st.write("※保存・出力は別途実装が必要です")
