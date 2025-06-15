import streamlit as st
import tempfile
import cv2
import numpy as np
import mediapipe as mp
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_video_with_pose(video_file_buffer):
    # アップロードされたファイルを一時的に保存
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file_buffer.read())
        temp_input_path = tfile.name

    cap = cv2.VideoCapture(temp_input_path)
    if not cap.isOpened():
        st.error("動画ファイルが開けません。ファイルが破損しているか、対応していない形式かもしれません。")
        # 一時ファイルをクリーンアップ
        os.unlink(temp_input_path)
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # FPSが不正な値の場合にデフォルトを設定
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    # 幅と高さを偶数に調整
    if width % 2 != 0:
        width -= 1
    if height % 2 != 0:
        height -= 1

    # 出力ファイルの一時パスを生成
    temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

    # mp4vでエラーが出る場合、'avc1'（H.264）を試す
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # または 'avc1'
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        st.error(f"出力ファイルを作成できません。コーデック '{'mp4v'}' がサポートされていない可能性があります。")
        cap.release()
        os.unlink(temp_input_path)
        # os.unlink(temp_output_path) # 作成失敗した場合は削除不要
        return None

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
            progress_bar.progress(min(1.0, (frame_count / total_frames))) # プログレスバーを更新

    cap.release()
    out.release()

    # 一時ファイルをクリーンアップ
    os.unlink(temp_input_path)

    if frame_count == 0:
        st.error("動画のフレームがありませんでした。")
        os.unlink(temp_output_path)
        return None

    # 処理された動画ファイルをバイナリで読み込み、戻り値とする
    with open(temp_output_path, 'rb') as f:
        processed_video_bytes = f.read()

    # 処理後のファイルをクリーンアップ
    os.unlink(temp_output_path)

    return processed_video_bytes

st.title("MediaPipe ポーズ特徴点動画生成")

uploaded_file = st.file_uploader("動画をアップロードしてください", type=["mp4", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.info("特徴点抽出中・・・しばらくお待ちください。")
    
    # スピナーを表示
    with st.spinner('動画を処理中...'):
        processed_video_bytes = process_video_with_pose(uploaded_file)
    
    if processed_video_bytes is not None:
        st.success("処理完了！特徴点動画を表示します。")
        st.video(processed_video_bytes)
    else:
        st.error("動画の処理に失敗しました。")
