import streamlit as st
import mediapipe as mp
import cv2
import tempfile
import os

st.title("📹 Mediapipe Pose WebApp")

video_file = st.file_uploader("🎞️ 動画をアップロードしてください", type=["mp4", "mov", "avi"])

if video_file is not None:
    # 一時ファイルへ保存（読み込み用）
    input_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_temp_file.write(video_file.read())

    # 出力ファイルのパスを定義
    output_temp_file = os.path.join(tempfile.gettempdir(), "output_processed.mp4")

    # 動画キャプチャ開始
    cap = cv2.VideoCapture(input_temp_file.name)
    if not cap.isOpened():
        st.error("❌ 動画を開けませんでした。")
    else:
        st.info("✅ 動画を読み込みました。Pose処理中...")

        # Mediapipe pose 初期化
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_temp_file, fourcc, fps, (width, height))

        with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                          min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                out.write(frame)

        cap.release()
        out.release()

        st.success("✅ 処理完了！再生はこちら👇")
        with open(output_temp_file, 'rb') as f:
            video_bytes = f.read()
            st.video(video_bytes)
