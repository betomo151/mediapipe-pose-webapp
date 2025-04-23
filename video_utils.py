import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        frames.append(frame)

    cap.release()
    return frames
