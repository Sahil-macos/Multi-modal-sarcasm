"""Central configuration for the multimodal sarcasm system."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("MSS_MODELS_DIR", ROOT / "models"))
FRONTEND_DIR = ROOT / "frontend"
UPLOAD_DIR = Path(os.environ.get("MSS_UPLOAD_DIR", ROOT / "uploads"))

ROBERTA_CKPT = MODELS_DIR / "roberta" / "roberta.pt"
HUBERT_DIR = MODELS_DIR / "hubert"
VIDEOMAE_DIR = MODELS_DIR / "videomae"

DEVICE = os.environ.get("MSS_DEVICE")  # None -> auto (cuda if available)

# Speech-to-text (faster-whisper). "tiny", "base", "small", ...
WHISPER_MODEL = os.environ.get("MSS_WHISPER_MODEL", "base")

# Video preprocessing
OCR_FRAMES = int(os.environ.get("MSS_OCR_FRAMES", 6))      # frames sampled for OCR
OCR_MIN_CONFIDENCE = 0.6
MAX_UPLOAD_MB = int(os.environ.get("MSS_MAX_UPLOAD_MB", 200))

# --- Fusion -----------------------------------------------------------------
# Prior reliability of each model (roughly its validation quality). Used as the
# base weight in late fusion; tune these if you have held-out numbers.
MODEL_WEIGHTS = {
    "text": 1.0,    # RoBERTa
    "audio": 0.9,   # HuBERT (val macro-F1 0.727)
    "video": 0.8,   # VideoMAE
}
# Decision thresholds per model (prob >= threshold -> sarcastic).
# HuBERT's threshold is read from its own config at load time.
TEXT_THRESHOLD = 0.5
VIDEO_THRESHOLD = 0.5

# --- Policy -----------------------------------------------------------------
FUSED_THRESHOLD = 0.5
UNCERTAIN_BAND = 0.08      # |p - threshold| below this -> UNCERTAIN
DISAGREEMENT_SPREAD = 0.45 # max-min modality prob above this -> incongruity flag
