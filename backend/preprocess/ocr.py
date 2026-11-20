"""On-screen text extraction (OCR) from video frames / images via RapidOCR."""
import difflib
import threading

import numpy as np

from .. import config

_engine = None
_lock = threading.Lock()


def _get_engine():
    global _engine
    with _lock:
        if _engine is None:
            from rapidocr import RapidOCR
            _engine = RapidOCR()
    return _engine


def _run(img: np.ndarray) -> list:
    """Return [(text, score)] for one RGB image, handling rapidocr API versions."""
    eng = _get_engine()
    bgr = np.ascontiguousarray(img[:, :, ::-1])
    out = eng(bgr)
    if hasattr(out, "txts"):  # rapidocr >= 2
        txts = out.txts or ()
        scores = out.scores or ()
        return list(zip(txts, scores))
    result = out[0] if isinstance(out, tuple) else out  # rapidocr_onnxruntime
    return [(r[1], float(r[2])) for r in (result or [])]


def ocr_image(img: np.ndarray) -> list:
    return [(t.strip(), float(s)) for t, s in _run(img)
            if t and t.strip() and float(s) >= config.OCR_MIN_CONFIDENCE]


def ocr_frames(frames: list, timestamps: list) -> dict:
    """OCR several frames, de-duplicating text that persists across frames."""
    lines, per_frame = [], []
    for img, ts in zip(frames, timestamps):
        found = ocr_image(img)
        per_frame.append({"t": ts, "lines": [t for t, _ in found]})
        for text, _ in found:
            if not any(difflib.SequenceMatcher(None, text.lower(), l.lower()).ratio() > 0.85
                       for l in lines):
                lines.append(text)
    return {"text": "\n".join(lines), "lines": lines, "frames": per_frame}
