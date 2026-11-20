"""Learned gated fusion over the three frozen sarcasm models.

Architecture: a Gated Multimodal Unit (Arevalo et al., 2017) extended to three
modalities with masking for missing inputs, plus a gated "expert score" path.

    per modality m in {text, audio, video}:
        x_m  = embedding the frozen model's own head sees  (768-d)
        z_m  = that model's threshold-calibrated logit
        h_m  = tanh(W_m · LN(x_m))                          (d-dim)

    gate   g = masked_softmax( W_g · [h_t, h_a, h_v, z, mask] )   (one weight per modality,
                                                                  missing modalities get 0)
    latent h = Σ g_m h_m
    logit    = w·h + b  +  α · Σ g_m z_m

The three backbone models are never updated; only this module (~300k params)
is trained, by scripts/train_gated_fusion.py. Modality dropout during training
teaches the gate to cope with text-only / audio-only / video-only inputs.
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MODALITIES = ("text", "audio", "video")


class GatedFusionNet(nn.Module):
    def __init__(self, dims=(768, 768, 768), d=128, dropout=0.3):
        super().__init__()
        self.dims, self.d = tuple(dims), d
        self.proj = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(k), nn.Dropout(dropout), nn.Linear(k, d), nn.Tanh())
            for k in dims
        ])
        n = len(dims)
        self.gate = nn.Sequential(nn.Linear(n * d + 2 * n, 64), nn.GELU(), nn.Linear(64, n))
        self.clf = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 1))
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, xs, z, mask):
        """xs: list of (B, dim_m); z, mask: (B, M). Returns logit (B,), gates (B, M), parts."""
        hs = torch.stack([p(x) for p, x in zip(self.proj, xs)], 1)        # B, M, d
        hs = hs * mask.unsqueeze(-1)
        zin = z * mask
        g_logits = self.gate(torch.cat([hs.flatten(1), zin, mask], -1))
        g_logits = g_logits.masked_fill(mask == 0, -1e4)
        g = g_logits.softmax(-1) * mask                                   # B, M
        h = (g.unsqueeze(-1) * hs).sum(1)
        latent = self.clf(h).squeeze(-1)
        expert = self.alpha * (g * zin).sum(-1)
        return latent + expert, g, {"latent": latent, "expert": expert}


def _logit(p, eps=1e-4):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


class GatedFusion:
    """Runtime wrapper: loads a trained gate and fuses per-request predictions."""

    def __init__(self, ckpt_path, device="cpu"):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        cfg = ck["config"]
        self.net = GatedFusionNet(cfg["dims"], cfg["d"], cfg.get("dropout", 0.3))
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval().to(device)
        self.device = device
        self.threshold = float(ck.get("decision_threshold", 0.5))
        self.metrics = ck.get("metrics", {})
        self.trained_on = ck.get("trained_on", "")
        self.path = str(ckpt_path)

    @torch.no_grad()
    def fuse(self, mod_inputs: dict) -> dict:
        """mod_inputs: {modality: {"embedding": np.ndarray, "z": float}} for present modalities."""
        xs, z, mask = [], [], []
        for m, dim in zip(MODALITIES, self.net.dims):
            if m in mod_inputs:
                xs.append(torch.as_tensor(np.asarray(mod_inputs[m]["embedding"], np.float32)).view(1, -1))
                z.append(mod_inputs[m]["z"]); mask.append(1.0)
            else:
                xs.append(torch.zeros(1, dim)); z.append(0.0); mask.append(0.0)
        xs = [x.to(self.device) for x in xs]
        zt = torch.tensor([z], device=self.device)
        mt = torch.tensor([mask], device=self.device)
        logit, g, parts = self.net(xs, zt, mt)
        p = float(torch.sigmoid(logit))
        # re-centre on the tuned decision threshold so 0.5 stays the boundary downstream
        p_centred = 1 / (1 + math.exp(-(_logit(p) - _logit(self.threshold))))
        return {
            "raw_probability": p,
            "probability": p_centred,
            "logit": float(logit),
            "gates": {m: float(g[0, i]) for i, m in enumerate(MODALITIES) if mask[i]},
            "latent_logit": float(parts["latent"]),
            "expert_logit": float(parts["expert"]),
            "alpha": float(self.net.alpha),
        }


def load_if_available(models_dir: Path):
    p = Path(models_dir) / "gated_fusion" / "gated_fusion.pt"
    return GatedFusion(p) if p.exists() else None
