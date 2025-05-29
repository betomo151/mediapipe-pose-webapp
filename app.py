import streamlit as st
import cv2
import tempfile
import time
import numpy as np

st.title("Streamlit Cloud対応！動画フレーム連続表示サンプル")

uploaded_file = st.file_uploader("MP4動画をアップロードしてください", type=["mp4"])

if uploaded_file is not None:
    # アップロード動画を一時ファイルに保存
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.flush()

    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("動画の読み込みに失敗しました")
    else:
        # FPS取得（なければ25fps固定）
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25
        delay = 1.0 / fps

        frame_placeholder = st.empty()  # フレーム表示用プレースホルダー

        st.info("動画処理中（実際はフレームを連続表示しています）...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # MediaPipeなどの処理をここに入れることが可能
            # 例：frame = your_mediapipe_process_function(frame)

            # BGR→RGB変換（OpenCVはBGR、StreamlitはRGB）
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 画面に表示
            frame_placeholder.image(frame)

            time.sleep(delay)  # 元動画のフレームレートで待機

        cap.release()
        st.success("動画の再生が終了しました。")
