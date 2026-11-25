"""Synthetic product surfaces: clean plates vs plates with scratch defects.

No external image corpus required — deterministic patterns for CI and demos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import IMG_SIZE, RANDOM_STATE, SYNTH_DIR


def _good_image(rng: np.random.Generator, size: int = IMG_SIZE) -> Image.Image:
    # Smooth light surface
    base = int(rng.integers(180, 220))
    noise = rng.normal(0, 4, size=(size, size, 3))
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _defect_image(rng: np.random.Generator, size: int = IMG_SIZE) -> Image.Image:
    img = _good_image(rng, size)
    draw = ImageDraw.Draw(img)
    # High-contrast black band — easy for a small CNN
    y = int(rng.integers(size // 4, (3 * size) // 4))
    draw.rectangle([(0, y - 2), (size - 1, y + 2)], fill=(0, 0, 0))
    # Optional second mark
    if rng.random() > 0.3:
        x = int(rng.integers(size // 4, (3 * size) // 4))
        draw.rectangle([(x - 2, 0), (x + 2, size - 1)], fill=(0, 0, 0))
    return img


def generate_split(
    n_per_class: int,
    split: str,
    out_dir: Path = SYNTH_DIR,
    seed: int = RANDOM_STATE,
    force: bool = False,
) -> list[tuple[Path, int]]:
    """Write PNGs and return list of (path, label). label 0=good, 1=defect."""
    rng = np.random.default_rng(seed + (0 if split == "train" else 99))
    records: list[tuple[Path, int]] = []
    for label, name, factory in (
        (0, "good", _good_image),
        (1, "defect", _defect_image),
    ):
        folder = out_dir / split / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            path = folder / f"{name}_{i:04d}.png"
            if force or not path.exists():
                factory(rng).save(path)
            records.append((path, label))
    return records


def default_transform(train: bool = False):
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
    if train:
        ops.insert(1, transforms.RandomHorizontalFlip())
    return transforms.Compose(ops)


class DefectDataset(Dataset):
    def __init__(self, records: list[tuple[Path, int]], train: bool = False):
        self.records = records
        self.transform = default_transform(train=train)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        path, label = self.records[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
