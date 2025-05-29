import streamlit as st
import tempfile
import cv2

def process_video(video_file):
    # 一時ファイルとして保存
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 互換性の高いコーデック
    out = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    if frame_count == 0:
        st.error("⚠️ フレームが書き出されませんでした。")
        return None

    with open(out_file.name, "rb") as f:
        return f.read()

st.title("🎥 動画処理テストアプリ")

uploaded_file = st.file_uploader("動画ファイルをアップロード（.mp4）", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.info("動画を処理中...")
    result_video = process_video(uploaded_file)
    if result_video:
        st.success("✅ 完了！")
        st.video(result_video)
