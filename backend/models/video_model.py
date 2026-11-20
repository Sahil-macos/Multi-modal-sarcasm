"""VideoMAE visual sarcasm classifier (models/videomae/).

The checkpoint ships no preprocessor config, so the standard VideoMAE
preprocessing is used: 16 uniformly sampled frames, shortest side resized to
224, centre crop 224x224, ImageNet mean/std normalisation.
"""
import threading

import numpy as np
import torch
from PIL import Image
from transformers import VideoMAEForVideoClassification

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VideoSarcasmDetector:
    name = "VideoMAE"
    modality = "video"

    def __init__(self, model_dir, device=None, threshold=0.5):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = VideoMAEForVideoClassification.from_pretrained(model_dir).to(self.device).eval()
        cfg = self.model.config
        self.num_frames = cfg.num_frames
        self.size = cfg.image_size
        self.id2label = {int(k): v for k, v in cfg.id2label.items()}
        self.pos = next(i for i, l in self.id2label.items() if l.lower().startswith("sarc"))
        self.threshold = threshold
        self._lock = threading.Lock()

    def _prep(self, frames):
        out = []
        for f in frames:
            im = Image.fromarray(f)
            w, h = im.size
            s = self.size / min(w, h)
            im = im.resize((max(self.size, round(w * s)), max(self.size, round(h * s))), Image.BILINEAR)
            w, h = im.size
            l, t = (w - self.size) // 2, (h - self.size) // 2
            im = im.crop((l, t, l + self.size, t + self.size))
            out.append((np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD)
        x = np.stack(out).transpose(0, 3, 1, 2)  # T, C, H, W
        return torch.from_numpy(x).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, frames: list, return_embedding: bool = False) -> dict:
        if len(frames) != self.num_frames:
            idx = np.linspace(0, len(frames) - 1, self.num_frames).round().astype(int)
            frames = [frames[i] for i in idx]
        captured = {}
        with self._lock:
            # capture exactly the vector the classification head receives
            hook = self.model.classifier.register_forward_hook(
                lambda m, inp, out: captured.__setitem__("h", inp[0].detach()))
            try:
                logits = self.model(pixel_values=self._prep(frames)).logits
            finally:
                hook.remove()
        p = float(logits.softmax(-1)[0, self.pos])
        extra = {"embedding": captured["h"][0].cpu().numpy()} if return_embedding else {}
        return extra | {
            "probability": round(p, 4),
            "label": "sarcastic" if p >= self.threshold else "not_sarcastic",
            "threshold": self.threshold,
            "frames_used": self.num_frames,
        }
