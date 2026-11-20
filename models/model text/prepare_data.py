from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA_FILE = Path("data/raw/MUStARD_Plus_Plus-main/mustard++_text.csv")
OUTPUT_DIR = Path("data/processed")
RANDOM_STATE = 42

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    required = {"SCENE", "SENTENCE", "Sarcasm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    print("Original rows:", len(df))

    # Keep only rows with known sarcasm labels.
    df = df.dropna(subset=["Sarcasm", "SENTENCE", "SCENE"]).copy()

    # Normalize labels and reject unexpected values.
    labels = pd.to_numeric(df["Sarcasm"], errors="coerce")
    df = df.loc[labels.notna()].copy()
    df["label"] = labels.loc[df.index].astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(
            "Expected binary labels 0 and 1. Found: "
            f"{sorted(df['label'].unique().tolist())}"
        )

    df["text"] = df["SENTENCE"].astype(str).str.replace(
        r"\s+", " ", regex=True
    ).str.strip()
    df = df[df["text"].ne("")].copy()

    # Keep one copy of duplicate scene/text/label records.
    df = df.drop_duplicates(subset=["SCENE", "text", "label"])
    df = df.reset_index(drop=True)

    print("\nLabeled examples:", len(df))
    print("Label counts:")
    print(df["label"].value_counts().sort_index().to_string())

    if df["label"].nunique() != 2:
        raise ValueError("Both sarcasm classes must be present.")

    # Split by scene to reduce leakage between related dialogue turns.
    first_split = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=RANDOM_STATE
    )
    train_val_idx, test_idx = next(
        first_split.split(df, groups=df["SCENE"])
    )
    train_val = df.iloc[train_val_idx].copy()
    test = df.iloc[test_idx].copy()

    second_split = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=RANDOM_STATE
    )
    train_idx, val_idx = next(
        second_split.split(train_val, groups=train_val["SCENE"])
    )
    train = train_val.iloc[train_idx].copy()
    val = train_val.iloc[val_idx].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, part in [
        ("train", train),
        ("validation", val),
        ("test", test),
    ]:
        part = part[["SCENE", "KEY", "text", "label"]].copy()
        part.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        print(
            f"\n{name}: {len(part)} rows; "
            f"labels={part['label'].value_counts().sort_index().to_dict()}"
        )

    # Verify that no scene appears in multiple splits.
    train_scenes = set(train["SCENE"])
    val_scenes = set(val["SCENE"])
    test_scenes = set(test["SCENE"])

    assert not train_scenes & val_scenes
    assert not train_scenes & test_scenes
    assert not val_scenes & test_scenes

    print("\nSUCCESS: processed CSV files created.")
    print("Scene leakage check passed.")

if __name__ == "__main__":
    main()
