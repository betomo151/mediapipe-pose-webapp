import streamlit as st
import mediapipe as mp
import cv2
import tempfile
import os
from io import BytesIO

st.title("📹 Mediapipe Pose WebApp")

video_file = st.file_uploader("🎞️ 動画をアップロードしてください", type=["mp4", "mov", "avi"])

if video_file is not None:
    # 読み込み用に一時ファイル保存
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_temp.write(video_file.read())
    input_temp.flush()

    # 出力用の一時ファイルパス
    output_path = os.path.join(tempfile.gettempdir(), "mediapipe_output.mp4")

    # 動画読み込み
    cap = cv2.VideoCapture(input_temp.name)

    if not cap.isOpened():
        st.error("❌ 動画を開けませんでした。")
    else:
        st.info("✅ Pose処理中...")

        # Mediapipe pose 初期化
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 書き込み先 VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        with mp_pose.Pose(static_image_mode=False,
                          model_complexity=1,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Mediapipeに渡して処理
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                out.write(frame)

        # ファイルを閉じる
        cap.release()
        out.release()

        st.success("✅ 処理完了！再生はこちら👇")

        # バイナリ読み込みして BytesIO 経由で表示
        with open(output_path, 'rb') as f:
            video_bytes = f.read()
            st.video(BytesIO(video_bytes))
