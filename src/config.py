from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SYNTH_DIR = DATA_DIR / "synthetic"
MODELS_DIR = BASE_DIR / "models"

for _d in (SYNTH_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "defect_cnn.pt"
METRICS_PATH = MODELS_DIR / "metrics.json"

CLASS_NAMES = ["good", "defect"]
IMG_SIZE = 32
RANDOM_STATE = 42

# Fast defaults for CPU / CI
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3
N_TRAIN_PER_CLASS = 200
N_TEST_PER_CLASS = 50
