from pathlib import Path
import time
import random
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

MODEL_NAME = "FacebookAI/roberta-base"
MAX_LENGTH = 128
SEED = 42

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("checkpoints/roberta_best")
METRICS_DIR = Path("outputs/metrics")
PLOTS_DIR = Path("outputs/plots")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = "Apple Silicon MPS"
else:
    DEVICE = "CPU"

print(f"Device available: {DEVICE}")
print(f"Model: {MODEL_NAME}")

def load_split(name):
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/prepare_data.py first."
        )

    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected labels in {path}")

    return df.reset_index(drop=True)

train_df = load_split("train")
val_df = load_split("validation")
test_df = load_split("test")

print(
    f"Examples: train={len(train_df)}, "
    f"validation={len(val_df)}, test={len(test_df)}"
)

class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer):
        self.encodings = tokenizer(
            dataframe["text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        self.labels = dataframe["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: torch.tensor(values[index], dtype=torch.long)
            for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(
            self.labels[index], dtype=torch.long
        )
        return item

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = SarcasmDataset(train_df, tokenizer)
val_dataset = SarcasmDataset(val_df, tokenizer)
test_dataset = SarcasmDataset(test_df, tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }

METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

args = TrainingArguments(
    output_dir="checkpoints/roberta_runs",
    num_train_epochs=5,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    warmup_steps=48,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
    seed=SEED,
    data_seed=SEED,
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2)
    ],
)

print("\nStarting RoBERTa fine-tuning...")
start_time = time.time()

train_result = trainer.train()

training_seconds = time.time() - start_time

print("\nSaving best model...")
trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))

print("\nEvaluating on the held-out test set...")
test_output = trainer.predict(test_dataset)
test_predictions = np.argmax(test_output.predictions, axis=-1)
test_labels = test_output.label_ids

precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0,
)

accuracy = accuracy_score(test_labels, test_predictions)

results = {
    "model": "RoBERTa-base",
    "best_validation_f1": trainer.state.best_metric,
    "test_accuracy": accuracy,
    "test_precision_macro": precision,
    "test_recall_macro": recall,
    "test_f1_macro": f1,
    "training_seconds": training_seconds,
    "train_examples": len(train_df),
    "validation_examples": len(val_df),
    "test_examples": len(test_df),
    "max_length": MAX_LENGTH,
    "seed": SEED,
}

pd.DataFrame([results]).to_csv(
    METRICS_DIR / "roberta_test_metrics.csv", index=False
)

pd.DataFrame({
    "text": test_df["text"],
    "true_label": test_labels,
    "predicted_label": test_predictions,
}).to_csv(
    METRICS_DIR / "roberta_test_predictions.csv", index=False
)

report = classification_report(
    test_labels,
    test_predictions,
    labels=[0, 1],
    target_names=["Not Sarcastic", "Sarcastic"],
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(report).transpose().to_csv(
    METRICS_DIR / "roberta_classification_report.csv"
)

pd.DataFrame(
    confusion_matrix(test_labels, test_predictions, labels=[0, 1]),
    index=["Actual_Not_Sarcastic", "Actual_Sarcastic"],
    columns=["Predicted_Not_Sarcastic", "Predicted_Sarcastic"],
).to_csv(METRICS_DIR / "roberta_confusion_matrix.csv")

print("\n===== FINAL TEST RESULTS =====")
for key, value in results.items():
    print(f"{key}: {value}")

print("\nSaved model:", MODEL_DIR)
print("Saved metrics:", METRICS_DIR)
print("\nTraining and evaluation completed successfully.")
