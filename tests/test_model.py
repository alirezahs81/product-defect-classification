from src.dataset import DefectDataset, generate_split, _defect_image, _good_image
from src.predict import DefectPredictor
from src.train import train
import numpy as np


def test_synthetic_images_differ():
    rng = np.random.default_rng(0)
    g = np.array(_good_image(rng))
    d = np.array(_defect_image(rng))
    assert g.shape == d.shape == (32, 32, 3)
    assert not np.allclose(g, d)


def test_dataset_len():
    rec = generate_split(5, "train", seed=1)
    assert len(rec) == 10
    ds = DefectDataset(rec)
    x, y = ds[0]
    assert x.shape[0] == 3
    assert y in (0, 1)


def test_train_and_predict():
    metrics = train(epochs=3, n_train=60, n_test=20)
    assert metrics["accuracy"] >= 0.7
    pred = DefectPredictor()
    # Use a generated defect image from disk if available
    rec = generate_split(2, "test", seed=7)
    defect_path = next(p for p, y in rec if y == 1)
    out = pred.predict_path(defect_path)
    assert out["label"] in {"good", "defect"}
    assert 0.0 <= out["confidence"] <= 1.0
