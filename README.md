# Multimodal Sarcasm Detector

One system that combines three sarcasm models (used as trained, with no retraining) behind a single API and web UI.

| Model | Modality | Checkpoint | Output |
|---|---|---|---|
| RoBERTa-base + MLP head | text | `models/roberta/roberta.pt` | P(sarcastic), threshold 0.5 |
| HuBERT-base + attention pooling (shipped package) | audio / speech prosody | `models/hubert/` | P(sarcastic), threshold 0.51 (from its config) |
| VideoMAE-base | video frames (face/gesture) | `models/videomae/` | P(Sarcastic), threshold 0.5 |

## Run

```
pip install -r requirements.txt
run.bat            # or: python -m uvicorn backend.app:app --port 8000
```
Open http://localhost:8000. The first start downloads the `roberta-base` tokenizer/config and Whisper `base` (~150 MB), then caches them.

## How an input flows

```
text / file ──► input_handler ──► MultimodalInput ──► pipeline ──► fusion ──► policy ──► JSON / UI
```

**Input handler** (`backend/preprocess/`) builds one structured record:

| Input | What is extracted | Goes to |
|---|---|---|
| Typed text | the text | RoBERTa |
| Document (PDF, DOCX, TXT/MD/CSV/JSON, HTML, images) | text; scanned PDFs and images go through OCR | RoBERTa |
| Audio (WAV, MP3, M4A, FLAC, OGG…) | 16 kHz waveform, plus a Whisper transcript | HuBERT, plus RoBERTa on the transcript |
| Video (MP4, MOV, MKV, WEBM…) | 16 uniformly sampled frames, the audio track, a transcript, and OCR of on-screen text from 6 frames (de-duplicated) | VideoMAE, HuBERT, plus RoBERTa on the transcript and the OCR text |

Text and a file can be sent together; they are analysed jointly.

**Fusion** (`backend/fusion.py`) uses threshold-calibrated, confidence-weighted late fusion:
1. Each score becomes a centred logit `z = logit(p) - logit(threshold)`, so models with different thresholds are comparable.
2. Text signals are merged into one text score. Each source is weighted by reliability (typed and document 1.0, transcript 0.85, OCR 0.6) and by length.
3. Modalities are averaged with `w = prior * (0.5 + confidence)`. Priors are in `config.MODEL_WEIGHTS`. A model far from its threshold counts more than one sitting on it.

**Gated fusion** (`backend/gated_fusion.py`, used automatically when `models/gated_fusion/gated_fusion.pt` exists):
a Gated Multimodal Unit over the three frozen models. It reads the embedding each model's own head sees, plus that model's calibrated logit, and a learned gate decides per input how much to trust text, audio and video. Missing modalities get zero weight. Only the gate (~300k parameters) is trained; the backbones are untouched.

Train or retrain it on MUStARD++:
```
python scripts/prepare_mustardpp.py --csv data/mustardpp/mustard++_text.csv --videos data/mustardpp/final_utterance_videos --out data/mustardpp/manifest.jsonl
python scripts/train_gated_fusion.py --manifest data/mustardpp/manifest.jsonl
```
Features are cached in `models/gated_fusion/features.npz`, so retraining only the gate takes minutes. Results are in `models/gated_fusion/metrics.json`.

Test results (MUStARD++, stratified random split 842/180/180, 180 test clips):

| | Accuracy | Macro-F1 | AUC |
|---|---|---|---|
| **Gated fusion** | **0.833** | **0.833** | 0.879 |
| Heuristic fusion | 0.800 | 0.799 | 0.897 |
| RoBERTa alone | 0.728 | 0.728 | 0.803 |
| HuBERT alone | 0.715 | 0.712 | 0.824 |
| VideoMAE alone | 0.676 | 0.676 | 0.741 |

Average gate weights on test: text 0.52, audio 0.36, video 0.12. **Caveat:** this split is not the one used to train the three backbones. If they saw some of these test clips during training, these numbers are optimistic. Pass the original split with `--split-file` for a clean evaluation.

**Policy engine** (`backend/policy.py`) outputs:
- a decision: `SARCASTIC`, `NOT_SARCASTIC`, `UNCERTAIN` (fused score within ±0.08 of 0.5) or `INSUFFICIENT_INPUT`
- a 0–100 score, a level, a confidence and the modality agreement
- flags: modality incongruity (one channel says sarcastic and another doesn't), short text or audio, a single modality only, preprocessing warnings, model errors
- a plain-language explanation of each model's contribution

All of these settings are in `backend/config.py`.

## API

`POST /api/analyze` (multipart): `text` (optional), `file` (optional), `transcribe` (bool), `ocr` (bool).
`GET /api/health`: model load status and fusion settings.

The response contains `input` (modalities, text segments, transcript with timestamps, OCR per frame, media info, warnings), `predictions` (one per model call), `fusion` (per-modality scores, weights, contributions), `result` (decision, score, flags, explanation) and `timings_s`.

## Notes and known limitations

- **RoBERTa pooling.** The checkpoint shipped without its inference script. The pooler output gives a near-constant score, while masked mean pooling separates sarcastic from literal text, so mean pooling is used (`POOLING` in `backend/models/text_model.py`). If your training notebook used `[CLS]` or the pooler, change that one constant.
- **VideoMAE preprocessing.** No `preprocessor_config.json` was included, so the standard VideoMAE preprocessing is used: 16 frames, resize, 224 centre crop, ImageNet mean/std.
- **HuBERT context.** The model supports a context clip, but the API currently runs it without one (zero context), as the original `predict()` does when no context is given.
- **Score spread.** The individual models' scores sit close to 0.5 on many inputs (HuBERT val macro-F1 is 0.73). Expect `UNCERTAIN` often. Tune `UNCERTAIN_BAND` and `MODEL_WEIGHTS` on held-out data.
- Everything runs on CPU. A 5 s video takes about 10–30 s (Whisper and OCR dominate).
