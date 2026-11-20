"""Speech-to-text with faster-whisper (CPU int8 by default)."""
import threading

import numpy as np

from .. import config

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel
            import torch
            cuda = torch.cuda.is_available() and config.DEVICE != "cpu"
            _model = WhisperModel(
                config.WHISPER_MODEL,
                device="cuda" if cuda else "cpu",
                compute_type="float16" if cuda else "int8",
            )
    return _model


def transcribe(wav: np.ndarray) -> dict:
    """Transcribe 16 kHz mono float32 audio. Returns text, language, segments."""
    if wav.size < 1600:  # < 0.1 s
        return {"text": "", "language": None, "segments": []}
    segments, info = _get_model().transcribe(
        wav, beam_size=5, vad_filter=True, condition_on_previous_text=False
    )
    segs = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments
    ]
    return {
        "text": " ".join(s["text"] for s in segs).strip(),
        "language": info.language,
        "language_probability": round(float(info.language_probability), 3),
        "segments": segs,
    }
