"""Build a gated-fusion training manifest from the MUStARD dataset.

    python scripts/prepare_mustard.py --json path/to/sarcasm_data.json \
        --videos path/to/utterances_final --out data/mustard/manifest.jsonl

sarcasm_data.json maps id -> {"utterance", "speaker", "context", "sarcasm", "show", ...};
utterance clips are <videos>/<id>.mp4. Split: stratified 70/15/15 (seed 42),
or speaker-independent with --split-by-show (test = FRIENDS, as in the paper).
"""
import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--videos", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split-by-show", action="store_true")
    args = ap.parse_args()

    data = json.loads(args.json.read_text(encoding="utf-8"))
    rows, missing = [], 0
    for k, v in data.items():
        vid = args.videos / f"{k}.mp4"
        if not vid.exists():
            missing += 1
            vid = None
        rows.append({"id": k, "label": int(bool(v["sarcasm"])), "text": v["utterance"],
                     "video": str(vid.resolve()) if vid else "", "show": v.get("show", "")})

    rng = random.Random(42)
    if args.split_by_show:
        for r in rows:
            r["split"] = "test" if r["show"] == "FRIENDS" else "train"
        tr = [r for r in rows if r["split"] == "train"]
        rng.shuffle(tr)
        for r in tr[: len(tr) // 7]:
            r["split"] = "val"
    else:
        for lab in (0, 1):
            g = [r for r in rows if r["label"] == lab]
            rng.shuffle(g)
            for i, r in enumerate(g):
                r["split"] = "train" if i < 0.7 * len(g) else "val" if i < 0.85 * len(g) else "test"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} items -> {args.out} ({missing} without video)")


if __name__ == "__main__":
    main()
