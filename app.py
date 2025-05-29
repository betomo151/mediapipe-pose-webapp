import streamlit as st
import tempfile
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_video_with_pose(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.flush()

    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("動画ファイルが開けません。")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps is None:
        fps = 25

    # 奇数を偶数に直す
    if width % 2 != 0:
        width -= 1
    if height % 2 != 0:
        height -= 1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # フレームリサイズ
            frame = cv2.resize(frame, (width, height))

            # MediaPipeはRGB画像を想定
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # 黒背景の真っ黒キャンバスを用意
            black_canvas = np.zeros_like(frame)

            if results.pose_landmarks:
                # 33個のランドマーク描画 (点と骨格線を描く)
                mp_drawing.draw_landmarks(
                    black_canvas, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=5, circle_radius=4),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
                )

            out.write(black_canvas)
            frame_count += 1

    cap.release()
    out.release()

    if frame_count == 0:
        st.error("動画のフレームがありませんでした。")
        return None

    with open(out_file.name, 'rb') as f:
        return f.read()

st.title("MediaPipe ポーズ特徴点動画生成")

uploaded_file = st.file_uploader("動画をアップロードしてください", type=["mp4", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.info("特徴点抽出中・・・")
    processed_video = process_video_with_pose(uploaded_file)
    if processed_video is not None:
        st.success("処理完了！特徴点動画を表示します。")
        st.video(processed_video)
