"""Fusion of per-model sarcasm predictions.

Two strategies, same output shape:

* **gated** (default when models/gated_fusion/gated_fusion.pt exists): a learned
  gate network (backend/gated_fusion.py) reads each frozen model's embedding and
  calibrated logit and decides, per input, how much to trust each modality.

* **heuristic** (fallback): threshold-calibrated, confidence-weighted average

      z_m = logit(p_m) - logit(t_m)
      w_m = prior_m * (0.5 + |2*sigmoid(z_m) - 1|)
      p   = sigmoid( Σ w_m z_m / Σ w_m )

In both, the several text signals (typed / document / transcript / OCR) are
first collapsed into one text modality, weighted by source reliability and
length; for the gated path their embeddings are averaged with the same weights.
"""
import math

import numpy as np

from . import config

TEXT_SOURCE_RELIABILITY = {"typed": 1.0, "document": 1.0, "transcript": 0.85, "ocr": 0.6}


def _logit(p, eps=1e-4):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + math.exp(-z))


def centred_logit(prob, threshold):
    return _logit(prob) - _logit(threshold)


def text_weights(sigs):
    return [TEXT_SOURCE_RELIABILITY.get(s["source"], 0.8) * min(1.0, 0.4 + 0.1 * s.get("n_words", 10))
            for s in sigs]


def collapse(signals: list, embeddings: dict = None) -> dict:
    """Group signals by modality -> {mod: {"z", "probability", "sources", "embedding"?}}."""
    by_mod = {}
    for i, s in enumerate(signals):
        s["z"] = centred_logit(s["probability"], s["threshold"])
        by_mod.setdefault(s["modality"], []).append((i, s))
    out = {}
    for mod, items in by_mod.items():
        sigs = [s for _, s in items]
        ws = text_weights(sigs) if mod == "text" else [1.0] * len(sigs)
        tot = sum(ws)
        for w, s in zip(ws, sigs):
            s["within_modality_weight"] = round(w / tot, 3)
        rec = {"z": sum(w * s["z"] for w, s in zip(ws, sigs)) / tot,
               "sources": [s["source"] for s in sigs]}
        rec["probability"] = _sigmoid(rec["z"])
        if embeddings is not None and all(i in embeddings for i, _ in items):
            rec["embedding"] = sum(w * embeddings[i] for w, (i, _) in zip(ws, items)) / tot
        out[mod] = rec
    return out


def _heuristic(mods):
    total_w = total = 0.0
    for mod, m in mods.items():
        conf = abs(2 * m["probability"] - 1)
        m["confidence"] = conf
        m["weight"] = config.MODEL_WEIGHTS.get(mod, 0.5) * (0.5 + conf)
        total_w += m["weight"]
        total += m["weight"] * m["z"]
    for m in mods.values():
        m["weight_share"] = m["weight"] / total_w
        m["logit_contribution"] = m["weight_share"] * m["z"]
    z = total / total_w
    return {"probability": _sigmoid(z), "logit": z,
            "method": "heuristic: threshold-calibrated, confidence-weighted late fusion"}


def _gated(mods, gate):
    r = gate.fuse({m: {"embedding": v["embedding"], "z": v["z"]} for m, v in mods.items()})
    for mod, m in mods.items():
        m["confidence"] = abs(2 * m["probability"] - 1)
        m["gate"] = r["gates"][mod]
        m["weight_share"] = r["gates"][mod]
        m["logit_contribution"] = r["alpha"] * r["gates"][mod] * m["z"]
    return {"probability": r["probability"], "logit": r["logit"],
            "method": "learned gated fusion (GMU over frozen model embeddings)",
            "gates": {k: round(v, 4) for k, v in r["gates"].items()},
            "gate_detail": {"latent_logit": round(r["latent_logit"], 4),
                            "expert_logit": round(r["expert_logit"], 4),
                            "alpha": round(r["alpha"], 4),
                            "raw_probability": round(r["raw_probability"], 4),
                            "decision_threshold": gate.threshold}}


def fuse(signals: list, embeddings: dict = None, gate=None) -> dict:
    """signals: [{"modality","source","model","probability","threshold","n_words"?}]
    embeddings: {signal_index: np.ndarray} (needed for gated fusion)."""
    if not signals:
        return {"probability": None, "modalities": {}, "contributions": [], "strategy": None}

    mods = collapse(signals, embeddings)
    use_gate = gate is not None and all("embedding" in m for m in mods.values())
    res = _gated(mods, gate) if use_gate else _heuristic(mods)
    res["strategy"] = "gated" if use_gate else "heuristic"
    if gate is None:
        res["strategy_note"] = "No trained gate found (models/gated_fusion/gated_fusion.pt); using heuristic fusion."

    contributions = []
    for mod, m in mods.items():
        m.pop("embedding", None)
        contributions.append({
            "modality": mod,
            "weight_share": round(m["weight_share"], 3),
            "logit_contribution": round(m["logit_contribution"], 3),
            "direction": "sarcastic" if m["z"] > 0 else "not_sarcastic",
        })
        for k in list(m):
            if isinstance(m[k], float):
                m[k] = round(m[k], 4)
    for s in signals:
        s["z"] = round(s["z"], 4)

    res.update(probability=round(res["probability"], 4), logit=round(res["logit"], 4),
               modalities=mods,
               contributions=sorted(contributions, key=lambda c: -abs(c["logit_contribution"])))
    return res
