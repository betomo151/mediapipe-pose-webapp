import streamlit as st
import tempfile
import cv2

def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.flush()

    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("動画の読み込みに失敗しました。")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 25  # fallback

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
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
        st.error("フレームが書き込まれませんでした。")
        return None

    with open(out_file.name, "rb") as f:
        return f.read()

# UI
st.title("MediaPipe Pose 動画処理 WebApp")

uploaded_file = st.file_uploader("動画をアップロードしてください", type=["mp4", "mov"])
if uploaded_file is not None:
    st.video(uploaded_file)
    st.info("ポーズ検出処理中...")
    result_video = process_video(uploaded_file)
    if result_video:
        st.success("✅ポーズ検出完了！以下の動画で確認できます。")
        st.video(result_video)  # または st.video(BytesIO(result_video))
