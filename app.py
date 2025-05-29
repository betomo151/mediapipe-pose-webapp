import streamlit as st
import tempfile
import cv2
import numpy as np
import mediapipe as mp

st.title("MediaPipe Pose 処理済み動画の生成と再生")

uploaded_file = st.file_uploader("MP4動画をアップロードしてください", type=["mp4"])

def process_and_save_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        st.error("動画の読み込みに失敗しました")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v'は汎用的でCloudでも動きやすい
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      enable_segmentation=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            annotated_frame = frame.copy()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
                )

            out.write(annotated_frame)
            frame_count += 1

    cap.release()
    out.release()

    if frame_count == 0:
        st.error("フレームが一つも処理されませんでした")
        return False
    return True

if uploaded_file is not None:
    # アップロード動画を一時ファイルに保存
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_temp.write(uploaded_file.read())
    input_temp.flush()

    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    with st.spinner("動画を処理中...少々お待ちください"):
        success = process_and_save_video(input_temp.name, output_temp.name)

    if success:
        st.success("処理済み動画の生成が完了しました。下の動画をご覧ください。")
        # 処理済み動画を読み込みStreamlitに渡す
        with open(output_temp.name, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
