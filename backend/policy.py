"""Policy engine: turns the fused score into a final decision + analysis."""
from . import config


def _level(p):
    if p >= 0.85:
        return "very high"
    if p >= 0.65:
        return "high"
    if p >= 0.5:
        return "moderate"
    if p >= 0.35:
        return "low"
    return "very low"


def decide(fused: dict, signals: list, mm_input) -> dict:
    flags, notes = [], []

    if fused.get("probability") is None:
        return {"decision": "INSUFFICIENT_INPUT", "sarcasm_score": None,
                "confidence": "none", "level": None,
                "flags": [{"code": "NO_SIGNAL", "severity": "high",
                           "message": "No model could score this input (no text, speech or video frames)."}],
                "explanation": ["Nothing analysable was extracted from the input."]}

    p = fused["probability"]
    t = config.FUSED_THRESHOLD
    margin = abs(p - t)

    # --- quality checks ------------------------------------------------------
    for s in signals:
        if s["modality"] == "text" and s.get("n_words", 99) < 3:
            flags.append({"code": "SHORT_TEXT", "severity": "low",
                          "message": f"{s['source']} text has fewer than 3 words; text score is unreliable."})
        if s["modality"] == "audio" and s.get("duration_s", 99) < 1.0:
            flags.append({"code": "SHORT_AUDIO", "severity": "medium",
                          "message": "Audio is under 1 second; prosody cues are limited."})
    for w in mm_input.warnings:
        flags.append({"code": "PREPROCESSING", "severity": "low", "message": w})

    mods = fused["modalities"]
    if len(mods) == 1:
        flags.append({"code": "SINGLE_MODALITY", "severity": "low",
                      "message": f"Only {next(iter(mods))} was available; no cross-modal confirmation."})

    # --- cross-modal incongruity (a classic sarcasm cue) --------------------
    probs = {m: v["probability"] for m, v in mods.items()}
    incongruent = False
    if len(probs) > 1:
        spread = max(probs.values()) - min(probs.values())
        dirs = {v > 0.5 for v in probs.values()}
        if spread >= config.DISAGREEMENT_SPREAD and len(dirs) > 1:
            incongruent = True
            hi = max(probs, key=probs.get)
            lo = min(probs, key=probs.get)
            flags.append({"code": "MODALITY_INCONGRUITY", "severity": "medium",
                          "message": f"{hi} says sarcastic ({probs[hi]:.2f}) but {lo} does not "
                                     f"({probs[lo]:.2f}). Mismatched channels often signal sarcasm, "
                                     f"but also lower confidence."})

    # --- decision -----------------------------------------------------------
    if margin < config.UNCERTAIN_BAND:
        decision = "UNCERTAIN"
    else:
        decision = "SARCASTIC" if p >= t else "NOT_SARCASTIC"

    agree = sum((v > 0.5) == (p >= t) for v in probs.values())
    if margin >= 0.3 and agree == len(probs) and len(probs) > 1:
        confidence = "high"
    elif margin >= 0.15 and not incongruent:
        confidence = "medium"
    else:
        confidence = "low"

    # --- explanation --------------------------------------------------------
    wname = "gate weight" if fused.get("strategy") == "gated" else "fused weight"
    names = {"text": "RoBERTa (text)", "audio": "HuBERT (audio)", "video": "VideoMAE (video)"}
    for c in fused["contributions"]:
        m = mods[c["modality"]]
        notes.append(f"{names.get(c['modality'], c['modality'])}: p={m['probability']:.2f} "
                     f"({c['direction'].replace('_', ' ')}), {c['weight_share']*100:.0f}% {wname}.")
    notes.append(f"Fused sarcasm probability {p:.2f} vs threshold {t:.2f} -> {decision.replace('_', ' ').lower()}"
                 f" ({agree}/{len(probs)} modalities agree).")

    return {
        "decision": decision,
        "sarcasm_score": round(p * 100, 1),
        "level": _level(p),
        "confidence": confidence,
        "agreement": f"{agree}/{len(probs)}",
        "flags": flags,
        "explanation": notes,
    }
