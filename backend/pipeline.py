"""Unified pipeline: MultimodalInput -> 3 models -> fusion -> policy."""
import threading
import time
import uuid

from . import config, fusion, policy
from .gated_fusion import load_if_available
from .preprocess.input_handler import MultimodalInput


class ModelRegistry:
    """Lazily loads each model once; a failed model is reported, not fatal."""

    def __init__(self):
        self._models, self._errors = {}, {}
        self._lock = threading.Lock()

    def _load(self, key):
        if key == "text":
            from .models.text_model import TextSarcasmDetector
            return TextSarcasmDetector(config.ROBERTA_CKPT, config.DEVICE, config.TEXT_THRESHOLD)
        if key == "audio":
            from .models.audio_model import AudioSarcasmDetector
            return AudioSarcasmDetector(config.HUBERT_DIR, config.DEVICE)
        if key == "video":
            from .models.video_model import VideoSarcasmDetector
            return VideoSarcasmDetector(config.VIDEOMAE_DIR, config.DEVICE, config.VIDEO_THRESHOLD)
        raise KeyError(key)

    def get(self, key):
        with self._lock:
            if key not in self._models and key not in self._errors:
                try:
                    self._models[key] = self._load(key)
                except Exception as e:
                    self._errors[key] = f"{type(e).__name__}: {e}"
        if key in self._errors:
            raise RuntimeError(self._errors[key])
        return self._models[key]

    def status(self):
        out = {}
        for k in ("text", "audio", "video"):
            out[k] = ("loaded" if k in self._models
                      else "error: " + self._errors[k] if k in self._errors else "not loaded")
        g = self._models.get("gate")
        out["gated_fusion"] = ("error: " + self._errors["gate"] if "gate" in self._errors
                               else "loaded" if g else "not trained (heuristic fusion)"
                               if "gate" in self._models else "not loaded")
        return out

    def gate(self):
        """Trained gated-fusion network, or None (falls back to heuristic fusion)."""
        with self._lock:
            if "gate" not in self._models and "gate" not in self._errors:
                try:
                    self._models["gate"] = load_if_available(config.MODELS_DIR)
                except Exception as e:
                    self._errors["gate"] = f"{type(e).__name__}: {e}"
        return self._models.get("gate")

    def reload_gate(self):
        with self._lock:
            self._models.pop("gate", None); self._errors.pop("gate", None)
        return self.gate()

    def warmup(self):
        for k in ("text", "audio", "video"):
            try:
                self.get(k)
            except Exception:
                pass
        self.gate()


registry = ModelRegistry()


def run(mm: MultimodalInput) -> dict:
    t0 = time.perf_counter()
    signals, predictions, errors, embs = [], [], [], []
    gate = registry.gate()
    want = gate is not None

    # ---- text model on every text segment ---------------------------------
    for seg in mm.text_segments:
        if not seg.text.strip():
            continue
        try:
            m = registry.get("text")
            ts = time.perf_counter()
            r = m.predict(seg.text, return_embedding=want)
            r.update(model=m.name, modality="text", source=seg.source,
                     n_words=len(seg.text.split()),
                     latency_s=round(time.perf_counter() - ts, 2),
                     input_preview=seg.text[:200])
            predictions.append(r)
        except Exception as e:
            errors.append({"model": "RoBERTa", "source": seg.source, "error": str(e)})

    # ---- audio model ----------------------------------------------------------
    if mm.waveform is not None and mm.waveform.size >= 1600:
        try:
            m = registry.get("audio")
            ts = time.perf_counter()
            r = m.predict(mm.waveform, return_embedding=want)
            r.update(model=m.name, modality="audio", source="audio_track" if mm.video_frames else "audio",
                     latency_s=round(time.perf_counter() - ts, 2))
            predictions.append(r)
        except Exception as e:
            errors.append({"model": "HuBERT", "error": str(e)})

    # ---- video model ----------------------------------------------------------
    if mm.video_frames:
        try:
            m = registry.get("video")
            ts = time.perf_counter()
            r = m.predict(mm.video_frames, return_embedding=want)
            r.update(model=m.name, modality="video", source="frames",
                     latency_s=round(time.perf_counter() - ts, 2))
            predictions.append(r)
        except Exception as e:
            errors.append({"model": "VideoMAE", "error": str(e)})

    for p in predictions:
        embs.append(p.pop("embedding", None))
        signals.append({k: p[k] for k in ("modality", "source", "model", "probability", "threshold")
                        if k in p} | {k: p[k] for k in ("n_words", "duration_s") if k in p})

    embeddings = {i: e for i, e in enumerate(embs) if e is not None}
    fused = fusion.fuse(signals, embeddings if want else None, gate)
    decision = policy.decide(fused, signals, mm)
    for e in errors:
        decision["flags"].append({"code": "MODEL_ERROR", "severity": "high",
                                  "message": f"{e['model']} failed: {e['error']}"})

    return {
        "id": uuid.uuid4().hex[:12],
        "input": mm.to_dict(),
        "predictions": predictions,
        "fusion": fused | {"signals": signals},
        "result": decision,
        "errors": errors,
        "timings_s": mm.timings | {"inference_total": round(time.perf_counter() - t0, 2)},
    }
