"""HuBERT audio sarcasm detector (models/hubert/).

The model class and `predict` logic are taken unchanged from the shipped
`sarcasm_audio_model.py` / `sarcasm_audio_detector.py`; only audio loading is
routed through our ffmpeg helper and the output is normalised for fusion.
"""
import json
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel

SAMPLE_RATE = 16000
LABELS = ["not_sarcastic", "sarcastic"]


class SarcasmAudioModel(nn.Module):
    def __init__(self, backbone=None, config=None, num_labels=2, dropout=0.2, use_context=False):
        super().__init__()
        if config is not None:
            config.layerdrop = 0.0
            self.backbone = AutoModel.from_config(config)
        else:
            self.backbone = AutoModel.from_pretrained(backbone, layerdrop=0.0)
        hidden = self.backbone.config.hidden_size
        n_layers = self.backbone.config.num_hidden_layers + 1
        self.use_context = use_context
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.attn = nn.Sequential(nn.Linear(hidden, 128), nn.Tanh(), nn.Linear(128, 1))
        in_dim = hidden * 3 if use_context else hidden
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_dim, 256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, num_labels),
        )

    def encode(self, input_values, input_lengths, attention_mask=None):
        out = self.backbone(input_values, attention_mask=attention_mask, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)
        w = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        x = (w.to(hs.dtype) * hs).sum(0)
        T = x.size(1)
        frames = self.backbone._get_feat_extract_output_lengths(input_lengths).clamp(min=1, max=T)
        mask = torch.arange(T, device=x.device)[None, :] < frames[:, None]
        scores = self.attn(x).squeeze(-1).float().masked_fill(~mask, -1e4)
        alpha = scores.softmax(-1).unsqueeze(-1).to(x.dtype)
        return (alpha * x).sum(1)

    def forward(self, input_values, input_lengths, attention_mask=None,
                ctx_values=None, ctx_lengths=None, ctx_attention_mask=None, ctx_present=None):
        u = self.encode(input_values, input_lengths, attention_mask)
        if not self.use_context:
            return self.head(u)
        if ctx_values is None:
            c = torch.zeros_like(u)
        else:
            c = self.encode(ctx_values, ctx_lengths, ctx_attention_mask)
            if ctx_present is not None:
                c = c * ctx_present.to(c.dtype)[:, None]
        return self.head(torch.cat([u, c, u * c], dim=-1))


class AudioSarcasmDetector:
    name = "HuBERT"
    modality = "audio"

    def __init__(self, model_dir, device=None):
        model_dir = Path(model_dir)
        with open(model_dir / "sarcasm_audio_config.json") as f:
            self.cfg = json.load(f)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fe = AutoFeatureExtractor.from_pretrained(model_dir)
        self.use_context = bool(self.cfg.get("use_context", False))
        self.model = SarcasmAudioModel(
            config=AutoConfig.from_pretrained(model_dir),
            dropout=self.cfg.get("dropout", 0.2), use_context=self.use_context,
        )
        state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.window = int(self.cfg.get("max_seconds", 10) * SAMPLE_RATE)
        self.ctx_keep = int(self.cfg.get("context_seconds", 8) * SAMPLE_RATE)
        self.threshold = float(self.cfg.get("threshold", 0.5))
        self._lock = threading.Lock()

    def _enc(self, arrs):
        e = self.fe(arrs, sampling_rate=SAMPLE_RATE, padding=True,
                    return_tensors="pt", return_attention_mask=True)
        am = e.get("attention_mask")
        return (e["input_values"].to(self.device),
                torch.tensor([len(a) for a in arrs], device=self.device),
                am.to(self.device) if am is not None else None)

    @torch.no_grad()
    def predict(self, wav: np.ndarray, context: np.ndarray = None,
                return_embedding: bool = False) -> dict:
        wav = np.asarray(wav, dtype=np.float32)
        hop = self.window // 2
        chunks = ([wav[i:i + self.window] for i in range(0, max(len(wav) - hop, 1), hop)]
                  if len(wav) > self.window else [wav])
        emb = None
        with self._lock:
            u = self._enc(chunks)
            used_ctx = False
            if return_embedding:  # utterance embedding = the pooled vector the head sees
                emb = self.model.encode(*u).mean(0).cpu().numpy()
            if self.use_context and context is not None and len(context):
                c = self._enc([context[-self.ctx_keep:]] * len(chunks))
                present = torch.ones(len(chunks), device=self.device)
                logits = self.model(*u, *c, present)
                used_ctx = True
            else:
                logits = self.model(*u)
        per = logits.softmax(-1)[:, 1].tolist()
        p = float(np.mean(per))
        extra = {"embedding": emb} if return_embedding else {}
        return extra | {
            "probability": round(p, 4),
            "label": LABELS[int(p >= self.threshold)],
            "threshold": self.threshold,
            "chunks": len(chunks),
            "chunk_probabilities": [round(x, 4) for x in per],
            "used_context": used_ctx,
            "duration_s": round(len(wav) / SAMPLE_RATE, 2),
        }
