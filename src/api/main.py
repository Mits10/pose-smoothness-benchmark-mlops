from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.smoothness import build_smoothness_features

app = FastAPI(title="Pose Smoothness Benchmark API", version="0.1.0")

HAND_KEYS = {"left_hand", "right_hand"}


class PredictRequest(BaseModel):
    sequence_id: str = Field(..., min_length=1)
    source: Literal["vicon", "xsens", "gopro"]
    fps: float = Field(..., gt=0)
    joints: dict[str, list[list[float]]]


class PredictResponse(BaseModel):
    sequence_id: str
    source: str
    predicted_reliability: float
    predicted_class: str
    smoothness: dict[str, float]
    alerts: list[str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not request.joints:
        raise HTTPException(status_code=400, detail="No hand keypoints provided.")

    bad_keys = set(request.joints) - HAND_KEYS
    if bad_keys:
        raise HTTPException(
            status_code=400,
            detail=f"Only hand keypoints are supported: {sorted(HAND_KEYS)}",
        )

    all_features: dict[str, float] = {}
    rms_values: list[float] = []

    for joint_name, coords in request.joints.items():
        signal = np.asarray(coords, dtype=float)
        if signal.ndim != 2 or signal.shape[1] not in (2, 3):
            raise HTTPException(
                status_code=400,
                detail=f"Joint '{joint_name}' must have shape (frames, 2|3).",
            )

        try:
            feats = build_smoothness_features(signal=signal, fps=request.fps)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        for key, value in feats.items():
            all_features[f"{joint_name}_{key}"] = value
        rms_values.append(feats["rms_jerk"])

    avg_rms = float(np.mean(rms_values))
    reliability = max(0.0, min(1.0, 1.0 / (1.0 + avg_rms)))
    predicted_class = "reliable" if reliability >= 0.5 else "noisy"

    alerts: list[str] = []
    if request.source != "vicon" and reliability < 0.5:
        alerts.append("Hand trajectory appears noisy relative to reference-like motion.")

    return PredictResponse(
        sequence_id=request.sequence_id,
        source=request.source,
        predicted_reliability=reliability,
        predicted_class=predicted_class,
        smoothness=all_features,
        alerts=alerts,
    )