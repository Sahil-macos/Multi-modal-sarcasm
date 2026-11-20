"""RoBERTa text sarcasm classifier (models/roberta/roberta.pt).

Checkpoint layout: {"model_state_dict", "model_name": "roberta-base",
"max_length": 128, "num_classes": 2}. The head is
classifier = Sequential(Linear(768,256), ReLU, Dropout, Linear(256,2)).

The original inference script was not shipped with the checkpoint. Probing
showed the pooler output gives a near-constant score, while attention-masked
mean pooling separates sarcastic from literal sentences, so POOLING="mean" is
the default. Change it here if your training code used a different pooling.
"""
import threading

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

POOLING = "mean"  # "mean" | "cls" | "pooler"
LABELS = ["not_sarcastic", "sarcastic"]


class RobertaSarcasmClassifier(nn.Module):
    def __init__(self, model_name="roberta-base", num_classes=2, dropout=0.3):
        super().__init__()
        self.roberta = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        hidden = self.roberta.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def pooled(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        if POOLING == "pooler":
            return out.pooler_output
        if POOLING == "cls":
            return out.last_hidden_state[:, 0]
        m = attention_mask.unsqueeze(-1).to(out.last_hidden_state.dtype)
        return (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, input_ids, attention_mask, return_embedding=False):
        h = self.pooled(input_ids, attention_mask)
        logits = self.classifier(h)
        return (logits, h) if return_embedding else logits


class TextSarcasmDetector:
    name = "RoBERTa"
    modality = "text"

    def __init__(self, ckpt_path, device=None, threshold=0.5):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model_name = ck.get("model_name", "roberta-base")
        self.max_length = int(ck.get("max_length", 128))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = RobertaSarcasmClassifier(self.model_name, int(ck.get("num_classes", 2)))
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.to(self.device).eval()
        self.threshold = threshold
        self._lock = threading.Lock()

    def _chunks(self, text):
        """Split long text into <=max_length token windows (50% overlap)."""
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        win = self.max_length - 2
        if len(ids) <= win:
            return [text]
        step = win // 2
        return [self.tok.decode(ids[i:i + win]) for i in range(0, len(ids) - step, step)]

    @torch.no_grad()
    def predict(self, text: str, return_embedding: bool = False) -> dict:
        chunks = self._chunks(text)
        with self._lock:
            enc = self.tok(chunks, padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            logits, emb = self.model(enc["input_ids"], enc["attention_mask"], return_embedding=True)
            probs = logits.softmax(-1)[:, 1]
        per = probs.tolist()
        # Sarcasm is often local: weight the most sarcastic window more.
        p = float(0.5 * max(per) + 0.5 * (sum(per) / len(per))) if len(per) > 1 else per[0]
        extra = {"embedding": emb.mean(0).cpu().numpy()} if return_embedding else {}
        return extra | {
            "probability": round(p, 4),
            "label": LABELS[int(p >= self.threshold)],
            "threshold": self.threshold,
            "chunks": len(chunks),
            "chunk_probabilities": [round(x, 4) for x in per],
        }
