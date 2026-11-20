"""Train the gated fusion network on top of the three FROZEN sarcasm models.

Usage
-----
    python scripts/train_gated_fusion.py --manifest data/mustard/manifest.jsonl

Manifest: one JSON object per line (or a CSV with the same columns)
    id      unique id
    label   1 = sarcastic, 0 = not
    text    utterance text            (optional)
    audio   path to audio file        (optional)
    video   path to video file        (optional; its audio track is used if `audio` is empty)
    split   train | val | test        (optional; stratified 70/15/15 if missing)
Relative paths are resolved against the manifest's folder.

Steps
  1. Run RoBERTa / HuBERT / VideoMAE once per item and cache the embedding each
     model's own head sees + its calibrated logit  (<out>/features.npz).
  2. Train GatedFusionNet (only ~300k params) with modality dropout.
  3. Tune the decision threshold on val; report test metrics for every single
     model, the heuristic fusion and the gated fusion; save the checkpoint to
     models/gated_fusion/gated_fusion.pt and metrics to metrics.json.
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.fusion import centred_logit  # noqa: E402
from backend.gated_fusion import MODALITIES, GatedFusionNet  # noqa: E402

DIM = 768


# --------------------------------------------------------------------------- data
def read_manifest(path: Path):
    if path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    base = path.parent
    for r in rows:
        r["label"] = int(r["label"])
        for k in ("audio", "video"):
            if r.get(k):
                p = Path(r[k])
                r[k] = str(p if p.is_absolute() else base / p)
    return rows


def assign_splits(rows, seed=42):
    if all(r.get("split") in ("train", "val", "test") for r in rows):
        return
    rng = random.Random(seed)
    for lab in (0, 1):
        grp = [r for r in rows if r["label"] == lab]
        rng.shuffle(grp)
        n = len(grp)
        for i, r in enumerate(grp):
            r["split"] = "train" if i < 0.7 * n else "val" if i < 0.85 * n else "test"


def extract(rows, out_dir: Path, transcribe_missing_text: bool):
    """Cache per-item embeddings + calibrated logits from the frozen models."""
    cache = out_dir / "features.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        if list(d["ids"]) == [str(r["id"]) for r in rows]:
            print(f"Using cached features: {cache}")
            return {k: d[k] for k in d.files}

    from backend.models.audio_model import AudioSarcasmDetector
    from backend.models.text_model import TextSarcasmDetector
    from backend.models.video_model import VideoSarcasmDetector
    from backend.preprocess import media

    print("Loading frozen models ...")
    tm = TextSarcasmDetector(config.ROBERTA_CKPT, config.DEVICE, config.TEXT_THRESHOLD)
    am = AudioSarcasmDetector(config.HUBERT_DIR, config.DEVICE)
    vm = VideoSarcasmDetector(config.VIDEOMAE_DIR, config.DEVICE, config.VIDEO_THRESHOLD)

    n = len(rows)
    X = {m: np.zeros((n, DIM), np.float32) for m in MODALITIES}
    Z = np.zeros((n, 3), np.float32)
    M = np.zeros((n, 3), np.float32)
    start = 0
    partial = out_dir / "features_partial.npz"
    if partial.exists():  # resume an interrupted extraction
        d = np.load(partial, allow_pickle=False)
        if list(d["ids"]) == [str(r["id"]) for r in rows]:
            start = int(d["done"])
            Z[:], M[:] = d["z"], d["mask"]
            for m in MODALITIES:
                X[m][:] = d[f"x_{m}"]
            print(f"Resuming feature extraction at {start}/{n}")
    t0 = time.time()
    for i, r in enumerate(rows):
        if i < start:
            continue
        text = (r.get("text") or "").strip()
        apath = r.get("audio") or r.get("video")
        try:
            if r.get("video"):
                frames = media.sample_frames(r["video"], vm.num_frames)
                o = vm.predict(frames, return_embedding=True)
                X["video"][i], Z[i, 2], M[i, 2] = o["embedding"], centred_logit(o["probability"], o["threshold"]), 1
        except Exception as e:
            print(f"  [{r['id']}] video failed: {e}")
        wav = None
        try:
            if apath:
                wav = media.load_audio(apath)
                if wav.size >= 1600:
                    o = am.predict(wav, return_embedding=True)
                    X["audio"][i], Z[i, 1], M[i, 1] = o["embedding"], centred_logit(o["probability"], o["threshold"]), 1
        except Exception as e:
            print(f"  [{r['id']}] audio failed: {e}")
        if not text and transcribe_missing_text and wav is not None and wav.size:
            from backend.preprocess.stt import transcribe
            text = transcribe(wav)["text"]
        if text:
            o = tm.predict(text, return_embedding=True)
            X["text"][i], Z[i, 0], M[i, 0] = o["embedding"], centred_logit(o["probability"], o["threshold"]), 1
        if (i + 1) % 25 == 0 or i + 1 == n:
            el = time.time() - t0
            k = i + 1 - start
            print(f"  {i + 1}/{n}  ({el:.0f}s, ~{el / k * (n - i - 1):.0f}s left)", flush=True)
            np.savez(partial, ids=np.array([str(r["id"]) for r in rows]), done=i + 1,
                     z=Z, mask=M, **{f"x_{m}": X[m] for m in MODALITIES})

    feats = {"ids": np.array([str(r["id"]) for r in rows]),
             "y": np.array([r["label"] for r in rows], np.float32),
             "split": np.array([r["split"] for r in rows]),
             "z": Z, "mask": M, **{f"x_{m}": X[m] for m in MODALITIES}}
    np.savez_compressed(cache, **feats)
    print(f"Saved features to {cache}")
    return feats


# --------------------------------------------------------------------------- metrics
def metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= thr).astype(int)
    f1s = []
    for c in (0, 1):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {"n": int(len(y)), "accuracy": round(float((pred == y).mean()), 4),
            "macro_f1": round(float(np.mean(f1s)), 4), "f1_sarcastic": round(f1s[1], 4),
            "auc": round(auc(y, p), 4)}


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum())
                 / (len(pos) * len(neg)))


def heuristic_probs(z, mask):
    out = []
    pri = np.array([config.MODEL_WEIGHTS[m] for m in MODALITIES])
    for zi, mi in zip(z, mask):
        if mi.sum() == 0:
            out.append(0.5); continue
        p = 1 / (1 + np.exp(-zi))
        w = pri * (0.5 + np.abs(2 * p - 1)) * mi
        out.append(float(1 / (1 + math.exp(-(w * zi).sum() / w.sum()))))
    return np.array(out)


def best_threshold(y, p):
    grid = np.linspace(0.2, 0.8, 61)
    return float(max(grid, key=lambda t: (metrics(y, p, t)["macro_f1"], -abs(t - 0.5))))


# --------------------------------------------------------------------------- training
def to_t(feats, idx):
    return ([torch.from_numpy(feats[f"x_{m}"][idx]) for m in MODALITIES],
            torch.from_numpy(feats["z"][idx]), torch.from_numpy(feats["mask"][idx]),
            torch.from_numpy(feats["y"][idx]))


def modality_dropout(mask, p):
    drop = (torch.rand_like(mask) < p).float() * mask
    new = mask - drop
    empty = new.sum(1) == 0
    new[empty] = mask[empty]  # always keep at least one present modality
    return new


@torch.no_grad()
def predict(net, xs, z, mask):
    net.eval()
    logit, g, _ = net(xs, z, mask)
    return torch.sigmoid(logit).numpy(), g.numpy()


def train(feats, args, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    tr = np.where(feats["split"] == "train")[0]
    va = np.where(feats["split"] == "val")[0]
    Xtr, Ztr, Mtr, Ytr = to_t(feats, tr)
    Xva, Zva, Mva, Yva = to_t(feats, va)
    net = GatedFusionNet((DIM,) * 3, args.hidden, args.dropout)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos = float(Ytr.sum()); neg = float(len(Ytr) - pos)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1)))
    best, best_state, bad = -1, None, 0
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(tr))
        for b in range(0, len(tr), args.batch_size):
            i = perm[b:b + args.batch_size]
            m = modality_dropout(Mtr[i], args.modality_dropout)
            logit, g, _ = net([x[i] for x in Xtr], Ztr[i], m)
            # small entropy bonus keeps the gate from collapsing onto one modality too early
            ent = -(g.clamp_min(1e-6).log() * g).sum(1).mean()
            loss = lossf(logit, Ytr[i]) - args.gate_entropy * ent
            opt.zero_grad(); loss.backward(); opt.step()
        p, _ = predict(net, Xva, Zva, Mva)
        f1 = metrics(Yva.numpy(), p)["macro_f1"]
        if f1 > best + 1e-4:
            best, bad = f1, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    net.load_state_dict(best_state)
    return net, best, ep + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=config.MODELS_DIR / "gated_fusion")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--modality-dropout", type=float, default=0.3)
    ap.add_argument("--gate-entropy", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, default=5, help="train N seeds, keep best on val")
    ap.add_argument("--transcribe-missing-text", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest)
    assign_splits(rows)
    print(f"{len(rows)} items: " + ", ".join(
        f"{s}={sum(r['split'] == s for r in rows)}" for s in ("train", "val", "test")))
    feats = extract(rows, args.out, args.transcribe_missing_text)
    print("modality coverage:", {m: int(feats['mask'][:, i].sum()) for i, m in enumerate(MODALITIES)})

    runs = []
    for s in range(args.seeds):
        net, f1, eps = train(feats, args, seed=s)
        print(f"  seed {s}: val macro-F1 {f1:.4f} ({eps} epochs)")
        runs.append((f1, s, net))
    best_f1, best_seed, net = max(runs, key=lambda r: r[0])

    va = np.where(feats["split"] == "val")[0]
    te = np.where(feats["split"] == "test")[0]
    if not len(te):
        te = va
    Xv, Zv, Mv, Yv = to_t(feats, va)
    thr = best_threshold(Yv.numpy(), predict(net, Xv, Zv, Mv)[0])

    Xt, Zt, Mt, Yt = to_t(feats, te)
    y = Yt.numpy()
    p_gate, g = predict(net, Xt, Zt, Mt)
    report = {"gated_fusion": metrics(y, p_gate, thr),
              "heuristic_fusion": metrics(y, heuristic_probs(Zt.numpy(), Mt.numpy()))}
    for i, m in enumerate(MODALITIES):
        sel = Mt[:, i].numpy() > 0
        if sel.any():
            report[f"{m}_only_model"] = metrics(y[sel], 1 / (1 + np.exp(-Zt[sel, i].numpy())))
    robustness = {}
    for i, m in enumerate(MODALITIES):  # gate with only one modality available
        sel = Mt[:, i].numpy() > 0
        if sel.any():
            mm = torch.zeros_like(Mt[sel]); mm[:, i] = 1
            robustness[f"gated_with_{m}_only"] = metrics(
                y[sel], predict(net, [x[sel] for x in Xt], Zt[sel], mm)[0], thr)
    mean_gates = {m: round(float(g[Mt[:, i].numpy() > 0, i].mean()), 4)
                  for i, m in enumerate(MODALITIES) if (Mt[:, i] > 0).any()}

    out = {"evaluated_on": "test" if len(np.where(feats["split"] == "test")[0]) else "val",
           "decision_threshold": thr, "best_seed": best_seed, "val_macro_f1": best_f1,
           "test": report, "robustness_missing_modalities": robustness,
           "mean_gate_weights_test": mean_gates, "hparams": vars(args) | {"manifest": str(args.manifest),
                                                                         "out": str(args.out)}}
    torch.save({"state_dict": net.state_dict(),
                "config": {"dims": [DIM] * 3, "d": args.hidden, "dropout": args.dropout},
                "decision_threshold": thr,
                "metrics": {k: v for k, v in out.items() if k != "hparams"},
                "trained_on": str(args.manifest.name)},
               args.out / "gated_fusion.pt")
    (args.out / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("decision_threshold", "test", "robustness_missing_modalities",
                                          "mean_gate_weights_test")}, indent=2))
    print(f"\nSaved {args.out / 'gated_fusion.pt'}")


if __name__ == "__main__":
    main()
