"""Nodes for the model_selection pipeline.

Cross-validates the candidate models on the TRAINING set only and selects the
champion by mean ROC-AUC. ROC-AUC is chosen over accuracy because the target is
imbalanced (~15.5% buyers); every estimator uses ``class_weight='balanced'`` so
the minority (buyer) class is not ignored.

The test set is NOT touched here -- CV on train is the validation mechanism.
This node outputs only the champion's *name + config* (a small dict), not a
fitted model, so model_train can do the final fit with a scaler that is fit on
train data alone (no leakage).
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)


def build_candidate(name: str, candidates: dict, selection: dict):
    """Instantiate a candidate estimator from params (shared with model_train).

    GradientBoosting has no ``class_weight`` arg, so imbalance is handled there
    via ROC-AUC selection + (optionally) sample weights at fit time elsewhere.
    """
    rs = selection["random_state"]
    cw = selection["class_weight"]
    cfg = candidates[name]

    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=cfg["max_iter"], C=cfg["C"], class_weight=cw, random_state=rs
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            class_weight=cw,
            random_state=rs,
            n_jobs=-1,
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            random_state=rs,
        )
    raise ValueError(f"Unknown candidate: {name}")


def select_champion(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidates: dict,
    selection: dict,
) -> dict:
    """Cross-validate candidates on train; return the ROC-AUC champion config.

    Returns:
        ``{"name": <champion>, "cv_scores": {...}, "best_score": float}``.
    """
    cv = StratifiedKFold(
        n_splits=selection["cv_folds"],
        shuffle=True,
        random_state=selection["random_state"],
    )

    cv_scores: dict[str, float] = {}
    for name in candidates:
        model = build_candidate(name, candidates, selection)
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring=selection["scoring"], n_jobs=-1
        )
        cv_scores[name] = float(scores.mean())
        logger.info(
            "CV %s: %s = %.4f (+/- %.4f)",
            name,
            selection["scoring"],
            scores.mean(),
            scores.std(),
        )

    champion = max(cv_scores, key=cv_scores.get)
    logger.info(
        "Champion = %s (%s = %.4f)",
        champion,
        selection["scoring"],
        cv_scores[champion],
    )
    return {
        "name": champion,
        "cv_scores": cv_scores,
        "best_score": cv_scores[champion],
    }
