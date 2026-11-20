
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEForVideoClassification
from tqdm import tqdm

DATA_FOLDER = "data/processed"
MODEL_FOLDER = "checkpoints/best_videomae"

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


class VideoDataset(Dataset):
    def __init__(self, indices):
        self.frames = np.load(
            os.path.join(DATA_FOLDER, "video_frames.npy"),
            mmap_mode="r"
        )

        self.labels = np.load(
            os.path.join(DATA_FOLDER, "labels.npy")
        )

        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]

        video = np.array(
            self.frames[idx],
            dtype=np.float32
        ) / 255.0

        video = torch.from_numpy(video).permute(0, 3, 1, 2)

        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)

        video = (video - mean) / std

        label = torch.tensor(
            int(self.labels[idx]),
            dtype=torch.long
        )

        return video, label


test_indices = np.load(
    os.path.join(DATA_FOLDER, "test_indices.npy")
)

test_dataset = VideoDataset(test_indices)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

model = VideoMAEForVideoClassification.from_pretrained(
    MODEL_FOLDER
)

model.to(device)
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for videos, labels in tqdm(
        test_loader,
        desc="Testing"
    ):

        videos = videos.to(device)

        outputs = model(pixel_values=videos)

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

tp = np.sum(
    (all_predictions == 1) & (all_labels == 1)
)

tn = np.sum(
    (all_predictions == 0) & (all_labels == 0)
)

fp = np.sum(
    (all_predictions == 1) & (all_labels == 0)
)

fn = np.sum(
    (all_predictions == 0) & (all_labels == 1)
)

accuracy = (tp + tn) / len(all_labels)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0

recall = tp / (tp + fn) if (tp + fn) > 0 else 0

f1_score = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0 else 0
)

print("\nTEST RESULTS")
print("--------------------------")
print("Test Samples:", len(all_labels))
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1_score:.4f}")

print("\nCONFUSION MATRIX")
print("--------------------------")
print("                 Predicted")
print("              Non-Sarc  Sarc")
print(f"Actual Non-Sarc   {tn:4d}    {fp:4d}")
print(f"Actual Sarc      {fn:4d}    {tp:4d}")
