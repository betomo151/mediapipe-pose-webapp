import streamlit as st
import mediapipe as mp
import cv2
import tempfile
import os
from io import BytesIO

st.set_page_config(layout="wide")
st.title("📹 Mediapipe Pose - 動画処理＆再生アプリ")

video_file = st.file_uploader("🎞️ 動画をアップロード（.mov, .mp4, .avi）", type=["mp4", "mov", "avi"])

if video_file is not None:
    # アップロード動画を一時ファイルに保存
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(video_file.read())
    temp_input.flush()

    # 出力用の一時ファイルパス（.mp4で出力）
    output_path = os.path.join(tempfile.gettempdir(), "output_processed.mp4")

    cap = cv2.VideoCapture(temp_input.name)
    if not cap.isOpened():
        st.error("❌ 動画を開けません。形式やコーデックを確認してください。")
    else:
        st.info("⏳ 動画処理中...")

        # 動画情報の取得
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # VideoWriterの初期化
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Mediapipe Poseの初期化
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                out.write(frame)

        cap.release()
        out.release()

        st.success("✅ 処理完了！以下で再生できます：")

        # 出力動画を読み込み、BytesIOに変換して再生
        with open(output_path, "rb") as f:
            video_bytes = f.read()
            st.video(BytesIO(video_bytes))
