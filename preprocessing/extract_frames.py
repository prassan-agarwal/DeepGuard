import cv2
import os
import numpy as np

def extract_frames(video_path, output_folder, num_frames=32):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(output_folder, exist_ok=True)

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int
    )

    saved_count = 0

    for idx in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret, frame = cap.read()

        if not ret:
            continue

        frame_path = os.path.join(
            output_folder,
            f"frame_{saved_count}.jpg"
        )

        cv2.imwrite(frame_path, frame)

        saved_count += 1

    cap.release()

    print(f"{saved_count} frames extracted from {video_path}")