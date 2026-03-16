import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm

# Initialize face detector
detector = MTCNN()

# number of frames to sample per video
NUM_FRAMES = 32

def extract_frames(video_path, num_frames=NUM_FRAMES):

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []

    if total_frames == 0:
        return frames

    indices = np.linspace(0, total_frames-1, num_frames, dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            continue

        frames.append(frame)

    cap.release()
    return frames


def detect_face(frame):

    faces = detector.detect_faces(frame)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]['box']

    x = max(0, x)
    y = max(0, y)

    face = frame[y:y+h, x:x+w]

    return face


def process_video(video_path, save_folder):

    frames = extract_frames(video_path)

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    count = 0

    for frame in frames:

        face = detect_face(frame)

        if face is None:
            continue

        face = cv2.resize(face, (224,224))

        filename = f"{video_name}_{count}.jpg"

        cv2.imwrite(os.path.join(save_folder, filename), face)

        count += 1


def process_dataset(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    videos = [v for v in os.listdir(input_folder) if v.endswith(".mp4")]

    for video in tqdm(videos):

        video_path = os.path.join(input_folder, video)

        process_video(video_path, output_folder)


if __name__ == "__main__":

    process_dataset(
        "dataset/raw/real",
        "dataset/processed/real"
    )

    process_dataset(
        "dataset/raw/fake",
        "dataset/processed/fake"
    )