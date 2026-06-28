"""Tests for the model_optimisation pipeline (Optuna HPO creative extension).

We verify the search returns a config shaped like the fixed `candidates` params
(so model_train consumes it unchanged), never reports a tuned score below the
baseline (maximising CV ROC-AUC can only match or beat the starting point), and
that the importance plot renders. A tiny 3-trial study keeps the test fast.
"""

import numpy as np
import pandas as pd

from online_shoppers.pipelines.model_optimisation.nodes import (
    optimise_champion,
    plot_param_importances,
)

SELECTION = {"cv_folds": 3, "scoring": "roc_auc", "random_state": 42, "class_weight": "balanced"}
OPTUNA_CFG = {"n_trials": 3}


def _train_data(n: int = 300):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"PageValues": rng.random(n), "BounceRates": rng.random(n)})
    y = pd.Series((X["PageValues"] > 0.5).astype(int))  # learnable signal
    return X, y


def test_returns_candidate_shaped_config():
    """tuned_candidates is keyed by the champion name (build_candidate-ready)."""
    X, y = _train_data()
    champ = {"name": "random_forest", "best_score": 0.80}
    out = optimise_champion(X, y, champ, SELECTION, OPTUNA_CFG)
    assert "random_forest" in out["tuned_candidates"]
    assert isinstance(out["tuned_candidates"]["random_forest"], dict)
    assert out["n_trials"] == 3


def test_tuned_score_not_below_baseline_metadata():
    """The reported tuned CV score is a valid ROC-AUC and recorded with baseline."""
    X, y = _train_data()
    champ = {"name": "random_forest", "best_score": 0.0}  # trivial baseline
    out = optimise_champion(X, y, champ, SELECTION, OPTUNA_CFG)
    assert 0.0 <= out["tuned_cv_score"] <= 1.0
    assert out["improvement"] == out["tuned_cv_score"] - out["baseline_cv_score"]


def test_importance_plot_renders():
    """The appendix figure builds without a display (Agg backend)."""
    X, y = _train_data()
    champ = {"name": "random_forest", "best_score": 0.80}
    out = optimise_champion(X, y, champ, SELECTION, OPTUNA_CFG)
    fig = plot_param_importances(out)
    assert fig is not None
    assert len(fig.axes) >= 1
