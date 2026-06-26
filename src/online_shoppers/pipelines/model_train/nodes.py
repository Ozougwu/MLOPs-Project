"""Nodes for the model_train pipeline (reqs #2 MLflow + #3 metrics & SHAP).

Final fit of the champion selected by model_selection:
  1. StandardScaler is fit on TRAIN ONLY, then applied to test -- the test set
     never influences any learned parameter (no leakage). Scaler + model are
     packaged together in one sklearn Pipeline so serving/prediction can't
     forget to scale.
  2. The sealed test set is evaluated ONCE here, with metrics appropriate to an
     imbalanced target: ROC-AUC, F1, precision/recall on the buyer class
     (accuracy is reported but not the headline).
  3. SHAP explains the model (req #3). A model-agnostic explainer is used so the
     champion can be linear or tree-based.

MLflow: ``autolog`` captures params/model/signature; we additionally log the
test metrics and the SHAP summary plot manually (autolog only sees CV/train).
The fitted Pipeline is returned and the catalog registers it as a model version.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # headless: no display in the Kedro run
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

from online_shoppers.pipelines.model_selection.nodes import build_candidate

logger = logging.getLogger(__name__)


def train_champion(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    champion_config: dict,
    candidates: dict,
    selection: dict,
) -> tuple[SkPipeline, dict]:
    """Fit champion (scaler train-only) and evaluate on the sealed test set.

    Returns:
        ``(fitted_pipeline, metrics_dict)``. The pipeline bundles the
        train-fit scaler + model so prediction always scales consistently.
    """
    name = champion_config["name"]
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # Scaler fit on TRAIN only, bundled with the model in one pipeline.
    model = build_candidate(name, candidates, selection)
    pipe = SkPipeline([("scaler", StandardScaler()), ("model", model)])

    mlflow.sklearn.autolog(log_models=False, silent=True)
    pipe.fit(X_train, y_train)

    # Evaluate ONCE on the sealed test set.
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    metrics = {
        "champion": name,
        "cv_roc_auc": champion_config["best_score"],
        "test_roc_auc": float(roc_auc_score(y_test, proba)),
        "test_f1_buyer": float(f1_score(y_test, pred)),
        "test_precision_buyer": float(precision_score(y_test, pred)),
        "test_recall_buyer": float(recall_score(y_test, pred)),
        "test_accuracy": float(accuracy_score(y_test, pred)),
    }
    mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
    logger.info(
        "Champion %s -> test ROC-AUC=%.4f, F1(buyer)=%.4f, recall(buyer)=%.4f",
        name,
        metrics["test_roc_auc"],
        metrics["test_f1_buyer"],
        metrics["test_recall_buyer"],
    )
    return pipe, metrics


def export_portable_model(pipe: SkPipeline) -> SkPipeline:
    """Pass the champion through to a portable local MLflow model directory.

    Identity node: the registry copy keeps versioning, while this writes a
    self-contained model dir (no absolute training paths) that serving and any
    clone can load with ``mlflow.sklearn.load_model(<dir>)``.
    """
    return pipe


def export_feature_columns(X_train: pd.DataFrame) -> dict:
    """Persist the exact trained feature schema for the serving API.

    Saving the column order alongside the model keeps serving and training in
    sync -- the API validates/reorders incoming rows against this list instead
    of hardcoding 33 names that could silently drift from the pipeline.
    """
    return {"feature_columns": list(X_train.columns)}


def explain_champion(
    pipe: SkPipeline, X_train: pd.DataFrame
) -> plt.Figure:
    """Produce a SHAP summary plot for the champion (req #3 explainability).

    Picks the explainer that matches the champion family -- ``TreeExplainer``
    for the tree models (exact and ~1000x faster than the permutation fallback),
    ``LinearExplainer`` for logistic regression -- so the plot stays cheap even
    on ~10k rows. SHAP runs on the SCALED feature space (the model's own input),
    with column names preserved so the plot is labelled correctly.
    """
    model = pipe.named_steps["model"]
    scaler = pipe.named_steps["scaler"]
    X_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=X_train.columns, index=X_train.index
    )
    sample = X_scaled.sample(min(1000, len(X_scaled)), random_state=42)

    model_name = model.__class__.__name__
    if model_name in ("RandomForestClassifier", "GradientBoostingClassifier"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        # Tree classifiers return per-class arrays; keep the buyer class (1).
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif getattr(shap_values, "ndim", 2) == 3:
            shap_values = shap_values[:, :, 1]
    elif model_name == "LogisticRegression":
        explainer = shap.LinearExplainer(model, sample)
        shap_values = explainer.shap_values(sample)
    else:  # safety net for any future candidate
        explainer = shap.Explainer(model.predict, sample)
        shap_values = explainer(sample).values

    fig = plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.title(f"SHAP summary — {model_name} (buyer class)")
    plt.tight_layout()
    logger.info("SHAP summary plot generated for %s.", model_name)
    return fig
