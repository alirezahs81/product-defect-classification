"""Train DefectCNN on synthetic good/defect images."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    METRICS_PATH,
    MODEL_PATH,
    N_TEST_PER_CLASS,
    N_TRAIN_PER_CLASS,
)
from src.dataset import DefectDataset, generate_split
from src.model import DefectCNN


def train(
    epochs: int = EPOCHS,
    n_train: int = N_TRAIN_PER_CLASS,
    n_test: int = N_TEST_PER_CLASS,
    device: str | None = None,
) -> dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_rec = generate_split(n_train, "train")
    test_rec = generate_split(n_test, "test")

    train_loader = DataLoader(
        DefectDataset(train_rec, train=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        DefectDataset(test_rec, train=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = DefectCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(yb)
            n += len(yb)
        history.append({"epoch": epoch, "train_loss": round(total_loss / max(n, 1), 4)})

    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().tolist()
            preds.extend(pred)
            ys.extend(yb.tolist())

    metrics = {
        "accuracy": round(float(accuracy_score(ys, preds)), 4),
        "precision": round(float(precision_score(ys, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(ys, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(ys, preds, zero_division=0)), 4),
        "n_test": len(ys),
        "epochs": epochs,
        "device": device,
        "history": history,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
