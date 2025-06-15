import streamlit as st
import tempfile
import cv2
import numpy as np
import mediapipe as mp
import os

# MediaPipe Poseモデルと描画ユーティリティを初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_video_with_pose(video_file_buffer):
    """
    アップロードされた動画からMediaPipeでポーズの特徴点を抽出し、
    黒い背景に特徴点のみを描画した動画を生成します。
    """
    # アップロードされたファイルを一時的に保存
    # withステートメントを使うことで、ブロック終了時にファイルが確実に閉じられる
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file_buffer.read())
        temp_input_path = tfile.name # 入力動画の一時パスを取得

    # 動画キャプチャオブジェクトの初期化
    cap = cv2.VideoCapture(temp_input_path)
    if not cap.isOpened():
        st.error("動画ファイルが開けません。ファイルが破損しているか、対応していない形式かもしれません。")
        os.unlink(temp_input_path) # エラー時にも一時ファイルを削除
        return None

    # 動画のプロパティを取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # FPSが不正な値の場合にデフォルトを設定
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    # 幅と高さを偶数に調整 (VideoWriterの要件に合わせるため)
    if width % 2 != 0:
        width -= 1
    if height % 2 != 0:
        height -= 1

    # 出力ファイルの一時パスを生成
    temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

    # VideoWriterの初期化
    # ここをH.264コーデックに変更します
    # 'avc1' (H.264) はウェブブラウザで広くサポートされています
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    # 他の一般的なコーデックの候補:
    # 'mp4v': MPEG-4 (一部環境では動作するが、より互換性の高いH.264を推奨)
    # 'XVID': Xvid (DivX互換)
    # 'MJPG': Motion JPEG (大きなファイルサイズになる傾向がある)

    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        st.error(f"出力ファイルを作成できません。コーデック '{'avc1'}' がサポートされていないか、システムにFFmpegが適切にインストールされていない可能性があります。")
        cap.release()
        os.unlink(temp_input_path)
        # temp_output_pathは作成失敗のため削除不要
        return None

    # MediaPipe Poseモデルのコンテキストマネージャ
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            total_frames = 1 # 0除算を防ぐ

        # 進捗バーの表示
        progress_text = st.empty() # 進捗テキスト用のプレースホルダー
        progress_bar = st.progress(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break # フレームの読み込みに失敗または動画の終わり

            # フレームのリサイズ
            frame = cv2.resize(frame, (width, height))

            # MediaPipeはRGB画像を想定しているため、BGRからRGBに変換
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # ポーズの特徴点描画用の黒いキャンバスを準備
            black_canvas = np.zeros_like(frame)

            if results.pose_landmarks:
                # 検出されたランドマークを黒いキャンバスに描画
                mp_drawing.draw_landmarks(
                    black_canvas,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    # ランドマーク（点）の描画設定
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=5, circle_radius=4),
                    # コネクション（骨格線）の描画設定
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
                )

            # 処理されたフレームを出力動画に書き込み
            out.write(black_canvas)
            frame_count += 1

            # 進捗バーとテキストを更新
            progress = min(1.0, (frame_count / total_frames))
            progress_bar.progress(progress)
            progress_text.text(f"処理中: {int(progress * 100)}%")

    # リソースを解放
    cap.release()
    out.release()

    # 入力一時ファイルを削除
    os.unlink(temp_input_path)

    if frame_count == 0:
        st.error("動画のフレームがありませんでした。処理する内容がありません。")
        os.unlink(temp_output_path) # 出力ファイルも削除
        return None

    # 処理された動画ファイルをバイナリモードで読み込み、その内容を返す
    with open(temp_output_path, 'rb') as f:
        processed_video_bytes = f.read()

    # 出力一時ファイルを削除
    os.unlink(temp_output_path)

    return processed_video_bytes

# Streamlitアプリケーションのメイン部分
st.title("MediaPipe ポーズ特徴点動画生成")

uploaded_file = st.file_uploader("動画をアップロードしてください", type=["mp4", "mov"])

if uploaded_file is not None:
    # アップロードされた動画をプレビュー表示
    st.video(uploaded_file)
    st.info("特徴点抽出中・・・しばらくお待ちください。")

    # 処理中のスピナーを表示
    with st.spinner('動画を処理中...'):
        processed_video_bytes = process_video_with_pose(uploaded_file)

    if processed_video_bytes is not None:
        st.success("処理完了！特徴点動画を表示します。")
        # 処理された動画のバイトデータをst.videoに渡して表示
        st.video(processed_video_bytes)
    else:
        st.error("動画の処理に失敗しました。")

