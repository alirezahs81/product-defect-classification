# Product Defect Classification

[![CI](https://github.com/alirezahs81/product-defect-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/alirezahs81/product-defect-classification/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Lightweight computer vision** for industrial-style **good vs defect** surface inspection.

Synthetic dataset (no external downloads) · small **CNN** (PyTorch) · accuracy / precision / recall / F1 · **FastAPI** image upload · **Docker** · **pytest** + GitHub Actions (CPU).

> Designed as a clean, portfolio-ready CV baseline: fast on CPU, deterministic in CI, and easy to swap for a real factory image folder later.

---

## What it does

| Stage | Detail |
|-------|--------|
| Data | Generates 32×32 RGB plates: clean texture (**good**) vs scratched/blobbed (**defect**) |
| Model | 3-block ConvNet → AdaptiveAvgPool → Linear(2) |
| Train | Adam + CrossEntropy; defaults tuned for CPU (~seconds) |
| Serve | `POST /predict` with multipart image → label + probabilities |

---

## Quick start

```bash
git clone https://github.com/alirezahs81/product-defect-classification.git
cd product-defect-classification
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py                 # train + print test metrics
pytest tests/ -v
uvicorn src.api:app --reload --port 8000
```

```bash
curl -s -F "file=@data/synthetic/test/defect/defect_0000.png" \
  http://localhost:8000/predict
```

### Docker

```bash
docker build -t defect-cnn .
docker run --rm -p 8000:8000 defect-cnn
```

---

## Layout

```text
product-defect-classification/
├── src/
│   ├── dataset.py    # synthetic image generation + Dataset
│   ├── model.py      # DefectCNN
│   ├── train.py
│   ├── predict.py
│   └── api.py
├── tests/
├── main.py
└── Dockerfile
```

---

## Methodology

1. **Synthesize** balanced good/defect images (scratches + dark spots on noisy plates).
2. **Train** a compact CNN with optional horizontal flip augmentation.
3. **Evaluate** held-out test accuracy / precision / recall / F1.
4. **Serve** via FastAPI; train-on-first-use if the checkpoint is missing.

On the synthetic test set, accuracy is typically **≥ 0.85** after a few CPU epochs (asserted ≥ 0.70 in tests).

---

## Extending to real data

Replace `generate_split` with a folder layout:

```text
data/real/train/good/*.jpg
data/real/train/defect/*.jpg
```

and point `DefectDataset` at those paths. Keep the same model and API contract.

---

## License

[MIT](LICENSE) © 2026 Alireza Hasheminasab
