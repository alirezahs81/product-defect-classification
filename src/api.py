"""FastAPI service for defect classification from uploaded images."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
import io

from src.config import MODEL_PATH
from src.predict import DefectPredictor
from src.train import train

app = FastAPI(
    title="Product Defect Classification",
    description="Lightweight CNN for good vs defect product surfaces.",
    version="1.0.0",
)

_predictor: DefectPredictor | None = None


def get_predictor() -> DefectPredictor:
    global _predictor
    if _predictor is None:
        if not MODEL_PATH.exists():
            train(epochs=3, n_train=80, n_test=20)
        try:
            _predictor = DefectPredictor()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _predictor


@app.get("/health")
def health() -> dict[str, Any]:
    ready = MODEL_PATH.exists()
    if not ready:
        return {"status": "model_missing", "model_ready": False}
    try:
        get_predictor()
        return {"status": "ok", "model_ready": True}
    except HTTPException:
        return {"status": "degraded", "model_ready": False}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="Expected an image upload")
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Invalid image: {exc}") from exc
    return get_predictor().predict_pil(image)
