import streamlit as st
import tempfile
import cv2

def process_video(video_file):
    # 一時ファイルに保存
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.flush()

    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("❌ 動画を読み込めませんでした。")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps > 240:
        fps = 25  # fallback

    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 再生互換性の高い形式
    out = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

    with open(out_file.name, "rb") as f:
        return f.read()

st.title("🎥 動画アップロード & 処理")

uploaded_file = st.file_uploader("動画ファイルをアップロードしてください（.mp4）", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.info("処理中...")
    processed = process_video(uploaded_file)
    if processed:
        st.success("✅ 処理完了！再生可能な動画：")
        st.video(processed)
    else:
        st.error("⚠️ 処理失敗：生成された動画が無効です。")
