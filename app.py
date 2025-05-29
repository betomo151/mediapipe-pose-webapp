import cv2

def convert_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("動画が開けません")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    # 偶数に丸め
    width = width if width % 2 == 0 else width - 1
    height = height if height % 2 == 0 else height - 1

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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

    print(f"書き出し完了: {frame_count} フレーム")
    return True

# 使い方例
convert_video('元動画.mp4', '出力動画.avi')
