import streamlit as st
import tempfile
import cv2
import mediapipe as mp
import numpy as np

st.title("MediaPipe Pose 処理済み動画作成＆再生（最小限サンプル）")

uploaded_file = st.file_uploader("動画アップロード", type=["mp4"])

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("動画が開けません")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25

    # 偶数に調整
    width = width if width % 2 == 0 else width - 1
    height = height if height % 2 == 0 else height - 1

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVIDは比較的互換性良し
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # サイズ調整（念のため）
            frame = cv2.resize(frame, (width, height))

            # RGBに変換して処理
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # ポーズ描画
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            out.write(frame)
            frame_count += 1

    cap.release()
    out.release()
    return frame_count > 0

if uploaded_file:
    # 一時ファイルに保存
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_temp.write(uploaded_file.read())
    input_temp.flush()

    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")  # 拡張子も合わせてみる

    with st.spinner("処理中..."):
        ok = process_video(input_temp.name, output_temp.name)

    if ok:
        st.success("処理完了！動画を再生します。")
        video_bytes = open(output_temp.name, "rb").read()
        st.video(video_bytes)
    else:
        st.error("動画処理に失敗しました。")
