import streamlit as st
import tempfile
import cv2
import numpy as np
import mediapipe as mp
import os

# MediaPipe Poseモデルと描画ユーティリティを初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_video_with_pose(video_file_buffer, target_resolution=None):
    """
    アップロードされた動画からMediaPipeでポーズの特徴点を抽出し、
    黒い背景に特徴点のみを描画した動画を生成します。
    target_resolution: (width, height)のタプルで指定された場合、その解像度で処理します。
                       Noneの場合、元の動画の解像度を使用します。
    """
    # アップロードされたファイルを一時的に保存
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file_buffer.read())
        temp_input_path = tfile.name # 入力動画の一時パスを取得
        st.write(f"デバッグ: 入力一時ファイルパス: {temp_input_path}") # デバッグ用

    # 動画キャプチャオブジェクトの初期化
    cap = cv2.VideoCapture(temp_input_path)
    if not cap.isOpened():
        st.error("動画ファイルが開けません。ファイルが破損しているか、対応していない形式かもしれません。")
        os.unlink(temp_input_path) # エラー時にも一時ファイルを削除
        st.write("デバッグ: cap.isOpened() が False でした。") # デバッグ用
        return None

    # 動画の元のプロパティを取得
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    st.write(f"デバッグ: 元の動画サイズ: {original_width}x{original_height}, FPS: {fps}") # デバッグ用

    # FPSが不正な値の場合にデフォルトを設定
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    # 処理する動画の幅と高さを決定
    if target_resolution:
        width, height = target_resolution
    else:
        width, height = original_width, original_height

    # 幅と高さを偶数に調整 (VideoWriterの要件に合わせるため)
    if width % 2 != 0:
        width -= 1
    if height % 2 != 0:
        height -= 1

    st.write(f"デバッグ: 処理対象サイズ: {width}x{height}, 最終FPS: {fps}") # デバッグ用

    # 出力ファイルの一時パスを生成
    temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    st.write(f"デバッグ: 出力一時ファイルパス: {temp_output_path}") # デバッグ用

    # VideoWriterの初期化
    # H.264コーデック ('avc1') を使用。Streamlit CloudではFFmpegのインストールが必須。
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    # もしH.264で動作しない場合は、以下のMJPGも試してください（ファイルサイズは大きくなります）
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        st.error(
            f"出力ファイルを作成できません。コーデック '{'avc1'}' がサポートされていないか、"
            "システムにFFmpegが適切にインストールされていない可能性があります。"
            "Streamlit Cloudのデプロイログをご確認ください。"
        )
        cap.release()
        os.unlink(temp_input_path)
        # temp_output_pathは作成失敗のため削除不要
        st.write(f"デバッグ: out.isOpened() が False でした。fourcc: {fourcc}") # デバッグ用
        return None

    # MediaPipe Poseモデルのコンテキストマネージャ
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            total_frames = 1 # 0除算を防ぐ

        progress_text = st.empty()
        progress_bar = st.progress(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 処理解像度に合わせてフレームをリサイズ
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
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=5, circle_radius=4),
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
    st.write("デバッグ: すべての一時ファイルを削除しました。") # デバッグ用

    return processed_video_bytes

# --- Streamlitアプリケーションのメイン部分 ---
st.title("MediaPipe ポーズ特徴点動画生成")

st.markdown("---")

st.sidebar.header("動画処理設定")
resolution_options = {
    "元の解像度": None, # 元の解像度で処理
    "640x360 (低画質/高速)": (640, 360),
    "854x480 (中画質/標準)": (854, 480),
    "1280x720 (高画質/低速)": (1280, 720)
}
selected_resolution_label = st.sidebar.selectbox(
    "処理解像度を選択:",
    options=list(resolution_options.keys()),
    index=1 # デフォルトを640x360に設定し、リソース消費を抑える
)
selected_resolution = resolution_options[selected_resolution_label]

st.warning("⚠️ Streamlit Cloudの無料枠にはリソース制限があります。処理が途中で止まる場合は、より低い「処理解像度」を選択するか、短い動画をお試しください。")
st.info("💡 **FFmpegのインストール確認**: GitHubリポジトリのルートに `packages.txt` ファイルを作成し、その中に `ffmpeg` と記述して再デプロイしているかご確認ください。これが解決策の鍵となります。")


uploaded_file = st.file_uploader("動画をアップロードしてください", type=["mp4", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.info("特徴点抽出中・・・しばらくお待ちください。")

    # 処理中のスピナーを表示
    with st.spinner('動画を処理中...'):
        # 選択された解像度を関数に渡す
        processed_video_bytes = process_video_with_pose(uploaded_file, selected_resolution)

    if processed_video_bytes is not None:
        st.success("処理完了！特徴点動画を表示します。")
        st.video(processed_video_bytes)
    else:
        st.error("動画の処理に失敗しました。詳細については、上記の警告やStreamlit Cloudのログをご確認ください。")

