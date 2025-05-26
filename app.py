import streamlit as st
import mediapipe as mp
import cv2
import tempfile
import numpy as np
from moviepy.editor import VideoFileClip

st.title("Mediapipe Pose WebApp")

video_file = st.file_uploader("🎥 動画をアップロードしてください（mov/mp4対応）", type=["mp4", "mov", "avi"])

if video_file is not None:
    # 一時ファイルとして保存
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    temp_input_path = tfile.name

    # 動画を読み込んでフレームごとに処理
    cap = cv2.VideoCapture(temp_input_path)

    if not cap.isOpened():
        st.error("❌ 動画を開けませんでした。形式やファイルを確認してください。")
    else:
        st.success("✅ 動画を読み込みました！ポーズ推定を実行中...")

        # Mediapipe初期化
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        # 出力用の一時動画ファイル
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = temp_output.name

        # 動画の情報を取得
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 書き込み用の VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # RGB変換して推定
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                # 描画
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # 出力
                out.write(frame)

        cap.release()
        out.release()

        st.success("✅ 処理完了！以下で再生できます：")
        st.video(output_path)
