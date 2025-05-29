import streamlit as st
import tempfile
import cv2

def process_video(video_file):
    # アップロードファイルを一時ファイルに保存
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.flush()

    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("動画が開けませんでした。ファイル形式を確認してください。")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps is None:
        fps = 25  # fps不明なら25に固定

    # 幅・高さは偶数に補正（奇数だと動画書き出しで失敗しやすい）
    if width % 2 != 0:
        width -= 1
    if height % 2 != 0:
        height -= 1

    # 動画コーデックは 'mp4v' or 'XVID' どちらかで試してね
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    if frame_count == 0:
        st.error("動画フレームがありませんでした。")
        return None

    with open(out_file.name, 'rb') as f:
        return f.read()

st.title("シンプル動画処理テスト")

uploaded_file = st.file_uploader("動画をアップロードしてください", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.info("動画処理中・・・")
    processed_video = process_video(uploaded_file)
    if processed_video is not None:
        st.success("動画処理完了！以下の動画を確認してください。")
        st.video(processed_video)
