"""Nodes for the model_predict pipeline (batch inference).

A first-class, separately-runnable pipeline (it is in the brief's diagram and
feeds both serving and the drift component). It loads the trained champion
Pipeline (scaler + model bundled) and scores a batch of feature rows, writing
predicted class + buyer probability to ``07_model_output``.

The champion Pipeline already contains the train-fit scaler, so this node must
NOT re-scale -- it just calls predict, exactly as the serving API will.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.pipeline import Pipeline as SkPipeline

logger = logging.getLogger(__name__)


def predict_batch(model: SkPipeline, batch: pd.DataFrame) -> pd.DataFrame:
    """Score a feature batch with the champion model.

    Args:
        model: trained champion Pipeline (scaler + classifier).
        batch: feature rows with the SAME columns the model was trained on.

    Returns:
        ``batch`` index + ``predicted_revenue`` + ``buyer_probability``.
    """
    proba = model.predict_proba(batch)[:, 1]
    pred = model.predict(batch)

    out = pd.DataFrame(
        {
            "predicted_revenue": pred.astype(int),
            "buyer_probability": proba,
        },
        index=batch.index,
    )
    logger.info(
        "Scored %d rows -> %d predicted buyers (%.1f%%).",
        len(out),
        int(out["predicted_revenue"].sum()),
        100 * out["predicted_revenue"].mean(),
    )
    return out
