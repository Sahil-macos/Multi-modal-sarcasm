"""Build a gated-fusion training manifest from MUStARD++.

    python scripts/prepare_mustardpp.py --csv path/to/mustard++_text.csv \
        --videos path/to/final_utterance_videos --out data/mustardpp/manifest.jsonl

MUStARD++ ships one CSV with both context and utterance rows
(KEY, SPEAKER, SENTENCE, SHOW, Sarcasm, Sarcasm_Type, ...). Utterance rows are
the ones with a Sarcasm label (their KEY usually ends in "_u"); context rows
("..._c_00", "..._c_01", ...) have it empty. Utterance clips are
<videos>/<KEY>.mp4 (searched recursively, any common video extension).

Column names are matched case-insensitively, so slightly different exports work.
Split: stratified 70/15/15 (seed 42); --split-by-show holds out one show as test.
If you already have the split used for training the three models, pass it with
--split-file (CSV or JSON: KEY -> train/val/test) so the gate is evaluated on
clips the backbones never saw.
"""
import argparse
import csv
import json
import random
from pathlib import Path

VIDEO_EXT = (".mp4", ".mkv", ".avi", ".mov", ".webm")


def col(header, *cands):
    low = {h.lower().strip(): h for h in header}
    for c in cands:
        if c in low:
            return low[c]
    for c in cands:
        for k, h in low.items():
            if c in k:
                return h
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--videos", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split-by-show", default=None, help="show name to hold out as test, e.g. FRIENDS")
    ap.add_argument("--split-file", type=Path, default=None)
    args = ap.parse_args()

    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        data = list(reader)
    k_key = col(header, "key", "id")
    k_txt = col(header, "sentence", "utterance", "text")
    k_lab = col(header, "sarcasm", "label")
    k_show = col(header, "show")
    if not (k_key and k_txt and k_lab):
        raise SystemExit(f"Could not find KEY/SENTENCE/Sarcasm columns in {header}")

    vids = {}
    for p in args.videos.rglob("*"):
        if p.suffix.lower() in VIDEO_EXT:
            vids.setdefault(p.stem, p.resolve())

    rows, missing = [], 0
    for r in data:
        lab = str(r.get(k_lab, "")).strip()
        if lab == "" or lab.lower() == "nan":
            continue  # context row
        key = r[k_key].strip()
        v = vids.get(key)
        missing += v is None
        rows.append({"id": key, "label": int(float(lab) > 0) if lab.replace(".", "").isdigit()
                     else int(lab.lower() in ("true", "yes", "sarcastic")),
                     "text": r[k_txt].strip(), "video": str(v) if v else "",
                     "show": r.get(k_show, "") if k_show else ""})

    rng = random.Random(42)
    if args.split_file:
        sp = (json.loads(args.split_file.read_text()) if args.split_file.suffix == ".json"
              else {a: b for a, b, *_ in csv.reader(open(args.split_file))})
        for r in rows:
            r["split"] = sp.get(r["id"], "train")
    elif args.split_by_show:
        for r in rows:
            r["split"] = "test" if r["show"].upper() == args.split_by_show.upper() else "train"
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
    pos = sum(r["label"] for r in rows)
    print(f"{len(rows)} utterances ({pos} sarcastic) -> {args.out}; "
          f"{len(rows) - missing} with video, {missing} without")


if __name__ == "__main__":
    main()
