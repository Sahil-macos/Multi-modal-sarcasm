
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEForVideoClassification
from tqdm import tqdm

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BATCH_SIZE = 1
EPOCHS = 12
PATIENCE = 3

BACKBONE_LR = 1e-5
CLASSIFIER_LR = 1e-3

MODEL_NAME = "MCG-NJU/videomae-base"
CHECKPOINT_DIR = "checkpoints/videomae_partial_finetune"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class VideoDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = frames
        self.labels = labels

        self.mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1)

        self.std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video = torch.tensor(
            self.frames[idx].copy(),
            dtype=torch.float32
        )

        video = video.permute(0, 3, 1, 2) / 255.0
        video = (video - self.mean) / self.std

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return video, label


frames = np.load(
    "data/processed/video_frames.npy",
    mmap_mode="r"
)

labels = np.load("data/processed/labels.npy")

train_indices = np.load(
    "data/processed/train_indices.npy"
)

val_indices = np.load(
    "data/processed/val_indices.npy"
)

train_dataset = VideoDataset(
    frames[train_indices],
    labels[train_indices]
)

val_dataset = VideoDataset(
    frames[val_indices],
    labels[val_indices]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


model = VideoMAEForVideoClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True
)

# Freeze the complete model first.
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the final two transformer blocks.
for layer in model.videomae.encoder.layer[-2:]:
    for param in layer.parameters():
        param.requires_grad = True

# Unfreeze the final layer normalization.
for param in model.videomae.layernorm.parameters():
    param.requires_grad = True

# Unfreeze the classification head.
for param in model.classifier.parameters():
    param.requires_grad = True

model.to(DEVICE)

backbone_params = [
    param
    for name, param in model.named_parameters()
    if param.requires_grad and not name.startswith("classifier.")
]

classifier_params = [
    param
    for name, param in model.named_parameters()
    if param.requires_grad and name.startswith("classifier.")
]

optimizer = torch.optim.AdamW(
    [
        {
            "params": backbone_params,
            "lr": BACKBONE_LR
        },
        {
            "params": classifier_params,
            "lr": CLASSIFIER_LR
        }
    ],
    weight_decay=0.01
)

criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0
patience_counter = 0

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} - Training"
    )

    for videos, targets in train_bar:

        videos = videos.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(pixel_values=videos)
        loss = criterion(outputs.logits, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

        train_correct += (
            predictions == targets
        ).sum().item()

        train_total += targets.size(0)

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for videos, targets in tqdm(
            val_loader,
            desc="Validation"
        ):

            videos = videos.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(pixel_values=videos)
            loss = criterion(outputs.logits, targets)

            val_loss += loss.item()

            predictions = torch.argmax(
                outputs.logits,
                dim=1
            )

            val_correct += (
                predictions == targets
            ).sum().item()

            val_total += targets.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:

        best_val_acc = val_acc
        patience_counter = 0

        model.save_pretrained(CHECKPOINT_DIR)

        print("Best model saved!")

    else:

        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("\nEarly stopping triggered.")
        break

print("\nTraining completed.")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"Model saved at: {CHECKPOINT_DIR}")
