"""FastAPI serving for the Online Shoppers champion model (req #4).

Loads the trained champion sklearn Pipeline (scaler + classifier bundled, so we
never re-scale here) and the canonical feature-column list, then exposes:
  - GET  /health  -> liveness  (process is up)
  - GET  /ready   -> readiness (model loaded) + the expected feature schema
  - POST /predict -> buyer probability + class for one or many feature rows

The request carries ALREADY-ENGINEERED feature rows (the 33 columns the model
was trained on, surfaced by /ready). This mirrors the model_predict pipeline
exactly, just over HTTP -- the feature-engineering pipeline owns raw->features,
the API owns features->prediction, with no duplicated transform logic.

Model source is configurable via env vars so the same image runs locally or in
a cluster. By default we load a PORTABLE local model directory (no absolute
training paths -> works in the container and any clone); the MLflow registry
alias is available as an opt-in alternative:
  MODEL_URI          (default: model/champion_model -- portable local dir;
                      set to e.g. models:/online_shoppers_champion/latest to
                      use the registry instead)
  FEATURE_COLUMNS    (default: model/feature_columns.json)
  MLFLOW_TRACKING_URI (default: ./mlruns; only needed for a models:/ URI)
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_URI = os.getenv("MODEL_URI", "model/champion_model")
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS", "model/feature_columns.json")

ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion model + feature schema once at startup."""
    # Only a registry alias (models:/...) needs tracking/registry URIs; a local
    # model directory loads directly. For a bare local mlruns path, normalize to
    # a proper file:// URI so the registry store accepts it.
    if MODEL_URI.startswith("models:/"):
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
        if "://" not in tracking_uri:
            tracking_uri = pathlib.Path(tracking_uri).resolve().as_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
    try:
        logger.info("Loading champion from %s", MODEL_URI)
        ml["model"] = mlflow.sklearn.load_model(MODEL_URI)
        with open(FEATURE_COLUMNS_PATH) as f:
            ml["columns"] = json.load(f)["feature_columns"]
        logger.info("Loaded model + %d feature columns.", len(ml["columns"]))
    except Exception as exc:  # boot anyway so /ready returns a clear 503
        logger.error("Model load failed: %s", exc)
    yield
    ml.clear()


app = FastAPI(
    title="Online Shoppers Purchase-Intent API",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    """A batch of already-engineered feature rows.

    Each row is a mapping of feature name -> value, matching the columns from
    GET /ready. Missing engineered columns default to 0 (e.g. absent one-hot).
    """

    rows: list[dict[str, float]] = Field(
        ..., description="Feature rows; keys must match the trained schema."
    )


def _align(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    """Build a DataFrame with exactly the trained columns, in order."""
    df = pd.DataFrame(rows)
    missing = [c for c in columns if c not in df.columns]
    for c in missing:  # tolerate absent one-hot / engineered cols -> 0
        df[c] = 0
    extra = [c for c in df.columns if c not in columns]
    if extra:
        raise HTTPException(
            status_code=422, detail=f"Unknown feature columns: {extra}"
        )
    return df[columns]


@app.post("/predict")
def predict(req: PredictRequest):
    if "model" not in ml:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not req.rows:
        raise HTTPException(status_code=422, detail="No rows provided.")

    X = _align(req.rows, ml["columns"])
    proba = ml["model"].predict_proba(X)[:, 1]
    pred = ml["model"].predict(X)
    return {
        "predictions": [
            {"predicted_revenue": int(p), "buyer_probability": float(pr)}
            for p, pr in zip(pred, proba)
        ]
    }


@app.get("/health")
def health():
    """Liveness: the process is running."""
    return {"status": "alive"}


@app.get("/ready")
def ready():
    """Readiness: model loaded; also returns the expected feature schema."""
    if "model" not in ml:
        raise HTTPException(status_code=503, detail="Model still initializing.")
    return {"status": "ready", "feature_columns": ml["columns"]}
