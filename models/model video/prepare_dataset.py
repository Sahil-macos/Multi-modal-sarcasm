
import os
import av
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

CSV_PATH = "data/raw/mustard++_text.csv"
VIDEO_FOLDER = "data/raw/video/final_utterance_videos"
OUTPUT_FOLDER = "data/processed"

NUM_FRAMES = 16
IMAGE_SIZE = 224


def extract_frames(video_path):
    container = av.open(video_path)

    decoded_frames = []

    for frame in container.decode(video=0):
        image = frame.to_ndarray(format="rgb24")
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        decoded_frames.append(image)

    container.close()

    if len(decoded_frames) == 0:
        raise ValueError("No frames could be decoded")

    indices = np.linspace(
        0,
        len(decoded_frames) - 1,
        NUM_FRAMES
    ).astype(int)

    selected_frames = [
        decoded_frames[index] for index in indices
    ]

    return np.array(selected_frames, dtype=np.uint8)


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df = df[df["Sarcasm"].notna()].reset_index(drop=True)

    processed_frames = []
    labels = []
    keys = []
    failed = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        key = str(row["KEY"])
        scene = str(row["SCENE"])

        filename = key + ".mp4"
        video_path = os.path.join(VIDEO_FOLDER, filename)

        if not os.path.exists(video_path):
            filename = scene + "_u.mp4"
            video_path = os.path.join(VIDEO_FOLDER, filename)

        if not os.path.exists(video_path):
            failed.append(filename)
            continue

        try:
            frames = extract_frames(video_path)

            processed_frames.append(frames)
            labels.append(int(row["Sarcasm"]))
            keys.append(key)

        except Exception as e:
            print("Error:", filename, e)
            failed.append(filename)

    if processed_frames:
        np.save(
            os.path.join(OUTPUT_FOLDER, "video_frames.npy"),
            np.stack(processed_frames)
        )

        np.save(
            os.path.join(OUTPUT_FOLDER, "labels.npy"),
            np.array(labels, dtype=np.int64)
        )

        np.save(
            os.path.join(OUTPUT_FOLDER, "keys.npy"),
            np.array(keys)
        )

    print("\nDataset preparation completed")
    print("Total labelled samples:", len(df))
    print("Processed videos:", len(processed_frames))
    print("Failed videos:", len(failed))
    print("Frames per video:", NUM_FRAMES)
    print("Frame dimensions:", IMAGE_SIZE, "x", IMAGE_SIZE)
    print("Labels:", len(labels))


if __name__ == "__main__":
    main()

