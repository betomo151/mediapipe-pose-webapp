import streamlit as st
import tempfile
import cv2
import os

def process_video(video_file):
    # 一時ファイルに保存
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.flush()
    input_path = tfile.name

    # 動画読み込み
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("❌ 動画の読み込みに失敗しました。")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps > 240 or fps != fps:  # NaN チェック
        fps = 25  # fallback

    # 出力ファイル
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_out.name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 互換性高め
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

    with open(output_path, 'rb') as f:
        return f.read()

st.title("✅ 動画処理テストアプリ")

uploaded_file = st.file_uploader("🎬 MP4動画をアップロードしてください", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.info("⏳ 処理中...（そのままお待ちください）")

    processed_video = process_video(uploaded_file)

    if processed_video:
        st.success("🎉 処理完了！以下が再生結果です。")
        st.video(processed_video)
    else:
        st.error("⚠️ 処理された動画が再生できませんでした。")
