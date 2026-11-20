
import os
import numpy as np
from sklearn.model_selection import train_test_split

DATA_FOLDER = "data/processed"

frames = np.load(os.path.join(DATA_FOLDER, "video_frames.npy"), mmap_mode="r")
labels = np.load(os.path.join(DATA_FOLDER, "labels.npy"))
keys = np.load(os.path.join(DATA_FOLDER, "keys.npy"))

indices = np.arange(len(labels))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    stratify=labels
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    stratify=labels[temp_idx]
)

os.makedirs(DATA_FOLDER, exist_ok=True)

np.save(os.path.join(DATA_FOLDER, "train_indices.npy"), train_idx)
np.save(os.path.join(DATA_FOLDER, "val_indices.npy"), val_idx)
np.save(os.path.join(DATA_FOLDER, "test_indices.npy"), test_idx)

print("Dataset splitting completed")
print("Training samples:", len(train_idx))
print("Validation samples:", len(val_idx))
print("Testing samples:", len(test_idx))

print("\nTraining label distribution:", np.bincount(labels[train_idx]))
print("Validation label distribution:", np.bincount(labels[val_idx]))
print("Testing label distribution:", np.bincount(labels[test_idx]))
