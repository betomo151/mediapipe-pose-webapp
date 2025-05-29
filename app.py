import streamlit as st
import cv2
import tempfile
import time
import numpy as np
import mediapipe as mp

st.title("MediaPipe Poseを使った動画フレーム処理＆再生")

uploaded_file = st.file_uploader("MP4動画をアップロードしてください", type=["mp4"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.flush()

    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("動画の読み込みに失敗しました")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25
        delay = 1.0 / fps

        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        frame_placeholder = st.empty()

        with mp_pose.Pose(static_image_mode=False,
                          model_complexity=1,
                          enable_segmentation=False,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:

            st.info("動画処理中... MediaPipe Poseで検出しています。")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # BGR→RGB変換（MediaPipeはRGB入力）
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # MediaPipeで姿勢検出
                results = pose.process(rgb_frame)

                # 結果を描画（OpenCVはBGRなので再変換）
                annotated_frame = frame.copy()
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
                    )

                # RGB変換（Streamlit用）
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # 画面に表示
                frame_placeholder.image(annotated_frame)

                time.sleep(delay)

        cap.release()
        st.success("動画処理と再生が終了しました。")
