import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from src.api import app
from src.dataset import _defect_image
from src.train import train


def setup_module():
    train(epochs=8, n_train=100, n_test=40)


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_ready"] is True


def test_predict_upload():
    client = TestClient(app)
    img = _defect_image(np.random.default_rng(0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    r = client.post(
        "/predict",
        files={"file": ("defect.png", buf, "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"good", "defect"}
    assert "probabilities" in body
