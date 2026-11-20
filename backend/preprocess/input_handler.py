"""Multimodal input handler.

Turns raw user input (typed text and/or an uploaded file) into one structured
representation, `MultimodalInput`, that the pipeline routes to the models:

    text   -> typed text segment
    doc    -> extracted text segment (PDF/DOCX/TXT/HTML, OCR for images/scans)
    audio  -> waveform (for HuBERT) + Whisper transcript (text segment)
    video  -> 16 frames (for VideoMAE) + audio track (HuBERT + transcript)
              + on-screen text via OCR (text segment)
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import media
from .documents import DOC_EXT

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus", ".webm_audio"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg", ".wmv"}


@dataclass
class TextSegment:
    source: str          # typed | document | transcript | ocr
    text: str

    def to_dict(self):
        return {"source": self.source, "text": self.text, "chars": len(self.text)}


@dataclass
class MultimodalInput:
    kind: str                                   # text | audio | video | document | mixed
    filename: Optional[str] = None
    text_segments: list = field(default_factory=list)
    waveform: Optional[np.ndarray] = None       # 16 kHz mono float32
    video_frames: Optional[list] = None         # 16 RGB frames for VideoMAE
    transcript: Optional[dict] = None
    ocr: Optional[dict] = None
    document: Optional[dict] = None
    media_info: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def modalities(self):
        m = []
        if any(s.text.strip() for s in self.text_segments):
            m.append("text")
        if self.waveform is not None and self.waveform.size:
            m.append("audio")
        if self.video_frames:
            m.append("video")
        return m

    def to_dict(self):
        """JSON-safe view (no raw arrays)."""
        return {
            "kind": self.kind,
            "filename": self.filename,
            "modalities": self.modalities,
            "text_segments": [s.to_dict() for s in self.text_segments],
            "transcript": self.transcript,
            "ocr": self.ocr,
            "document": ({k: v for k, v in self.document.items() if k != "text"}
                         if self.document else None),
            "media": self.media_info,
            "timings_s": self.timings,
            "warnings": self.warnings,
        }


def classify_file(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in VIDEO_EXT:
        return "video"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in DOC_EXT:
        return "document"
    raise ValueError(
        f"Unsupported file type '{ext}'. Supported: "
        f"video {sorted(VIDEO_EXT)}, audio {sorted(AUDIO_EXT)}, documents {sorted(DOC_EXT)}"
    )


def _timed(rec: MultimodalInput, key: str, fn, *a, **kw):
    t = time.perf_counter()
    try:
        return fn(*a, **kw)
    finally:
        rec.timings[key] = round(time.perf_counter() - t, 2)


def _handle_audio_track(rec: MultimodalInput, path, transcribe: bool):
    from .stt import transcribe as stt
    wav = _timed(rec, "decode_audio", media.load_audio, path)
    if wav.size == 0:
        rec.warnings.append("No audio track found.")
        return
    rec.waveform = wav
    rec.media_info["audio_duration_s"] = round(wav.size / media.SAMPLE_RATE, 2)
    if float(np.abs(wav).max()) < 1e-4:
        rec.warnings.append("Audio track is silent.")
    if transcribe:
        try:
            rec.transcript = _timed(rec, "speech_to_text", stt, wav)
            if rec.transcript["text"]:
                rec.text_segments.append(TextSegment("transcript", rec.transcript["text"]))
            else:
                rec.warnings.append("No speech detected in audio.")
        except Exception as e:
            rec.warnings.append(f"Speech-to-text failed: {e}")


def build_input(text: Optional[str] = None, file_path=None, filename: Optional[str] = None,
                transcribe: bool = True, ocr: bool = True,
                video_num_frames: int = 16) -> MultimodalInput:
    kinds = []
    rec = MultimodalInput(kind="text", filename=filename)

    if text and text.strip():
        rec.text_segments.append(TextSegment("typed", text.strip()))
        kinds.append("text")

    if file_path is not None:
        fkind = classify_file(filename or str(file_path))
        kinds.append(fkind)

        if fkind == "document":
            from .documents import extract_document
            doc = _timed(rec, "document_extraction", extract_document, file_path)
            rec.document = doc
            if doc["text"]:
                rec.text_segments.append(TextSegment("document", doc["text"]))
            else:
                rec.warnings.append("No text could be extracted from the document.")

        elif fkind == "audio":
            _handle_audio_track(rec, file_path, transcribe)

        elif fkind == "video":
            info = media.video_info(file_path)
            rec.media_info.update({"video": info})
            rec.video_frames = _timed(rec, "frame_sampling", media.sample_frames,
                                      file_path, video_num_frames)
            _handle_audio_track(rec, file_path, transcribe)
            if ocr:
                from .. import config
                from .ocr import ocr_frames
                n = config.OCR_FRAMES
                frames = media.sample_frames(file_path, n)
                dur = info.get("duration_s") or 0
                ts = [round(float(x), 2) for x in np.linspace(0, dur, n)]
                try:
                    rec.ocr = _timed(rec, "ocr", ocr_frames, frames, ts)
                    if rec.ocr["text"]:
                        rec.text_segments.append(TextSegment("ocr", rec.ocr["text"]))
                except Exception as e:
                    rec.warnings.append(f"OCR failed: {e}")

    if not kinds:
        raise ValueError("Provide text and/or a file.")
    rec.kind = kinds[0] if len(kinds) == 1 else "mixed"
    return rec
