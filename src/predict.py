"""Inference helpers for single images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.config import CLASS_NAMES, MODEL_PATH
from src.dataset import default_transform
from src.model import DefectCNN


class DefectPredictor:
    def __init__(self, model_path: Path = MODEL_PATH, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Run: python -m src.train"
            )
        payload = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model = DefectCNN()
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.transform = default_transform(train=False)

    def predict_pil(self, image: Image.Image) -> dict[str, Any]:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
            pred = int(torch.argmax(logits, dim=1).item())
        return {
            "label": CLASS_NAMES[pred],
            "class_id": pred,
            "confidence": round(float(max(probs)), 4),
            "probabilities": {
                CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)
            },
        }

    def predict_path(self, path: Path | str) -> dict[str, Any]:
        img = Image.open(path).convert("RGB")
        return self.predict_pil(img)
