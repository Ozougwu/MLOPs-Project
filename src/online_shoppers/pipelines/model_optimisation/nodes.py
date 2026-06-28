"""model_optimisation nodes (creative extension): Optuna hyperparameter search.

The required path selects a champion *family* by cross-validated ROC-AUC with
fixed, modest hyperparameters (the `candidates` params). This optional stage
asks a sharper question: within that champion family, what hyperparameters
maximise CV ROC-AUC on the training set? We use Optuna's TPE sampler -- a
Bayesian search that concentrates trials in promising regions, far more
sample-efficient than grid search over the same space.

Design choices that keep this honest and reproducible:
  - The search optimises the SAME metric and CV scheme as model_selection
    (5-fold stratified ROC-AUC on train only) -- the test set is never seen,
    so there is no leakage and no optimistic bias in the reported gain.
  - The sampler is seeded, so a given study is reproducible.
  - Output is a `tuned_candidates` dict with the SAME shape as the fixed
    `candidates` params, so model_train consumes it unchanged -- the extension
    plugs in without altering the required pipeline's code.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # headless: no display in CI / containers
import matplotlib.pyplot as plt
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)  # quiet per-trial spam


def _suggest_params(trial: optuna.Trial, name: str, selection: dict) -> dict:
    """Define the per-family search space and build an estimator for one trial.

    Spaces are deliberately wider than the fixed candidate defaults so the
    search can actually improve on them, but bounded to sensible ranges for a
    ~10k-row tabular problem (avoiding overfit-prone extremes).
    """
    rs = selection["random_state"]
    cw = selection["class_weight"]

    if name == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        model = RandomForestClassifier(
            **params, class_weight=cw, random_state=rs, n_jobs=-1
        )
    elif name == "gradient_boosting":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        model = GradientBoostingClassifier(**params, random_state=rs)
    elif name == "logistic_regression":
        params = {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "max_iter": 1000,
        }
        model = LogisticRegression(**params, class_weight=cw, random_state=rs)
    else:
        raise ValueError(f"Unknown champion family for HPO: {name}")

    return params, model


def optimise_champion(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    champion_config: dict,
    selection: dict,
    optuna_cfg: dict,
) -> dict:
    """Optuna TPE search over the champion family; return tuned candidate params.

    Returns a dict shaped like the `candidates` params (so model_train reads it
    unchanged), plus the tuned CV score and the per-trial history for the report.
    """
    name = champion_config["name"]
    baseline = champion_config["best_score"]
    cv = StratifiedKFold(
        n_splits=selection["cv_folds"],
        shuffle=True,
        random_state=selection["random_state"],
    )

    def objective(trial: optuna.Trial) -> float:
        _, model = _suggest_params(trial, name, selection)
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring=selection["scoring"], n_jobs=-1
        )
        return float(scores.mean())

    sampler = optuna.samplers.TPESampler(seed=selection["random_state"])
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=optuna_cfg["n_trials"], show_progress_bar=False)

    tuned = study.best_params.copy()
    tuned.setdefault("max_iter", 1000)  # logreg needs it; harmless elsewhere
    improvement = study.best_value - baseline
    logger.info(
        "Optuna HPO (%s, %d trials): CV %s %.4f -> %.4f (%+.4f) ",
        name,
        optuna_cfg["n_trials"],
        selection["scoring"],
        baseline,
        study.best_value,
        improvement,
    )

    # Shape like `candidates` so model_train's build_candidate consumes it as-is.
    tuned_candidates = {name: tuned}
    return {
        "tuned_candidates": tuned_candidates,
        "champion": name,
        "baseline_cv_score": baseline,
        "tuned_cv_score": float(study.best_value),
        "improvement": float(improvement),
        "n_trials": optuna_cfg["n_trials"],
        "best_params": study.best_params,
        # importances need >1 distinct param; guard for trivial spaces
        "param_importances": _safe_importances(study),
    }


def _safe_importances(study: optuna.Study) -> dict:
    """Optuna param importances, or {} if they can't be computed."""
    try:
        return {k: float(v) for k, v in optuna.importance.get_param_importances(study).items()}
    except Exception as exc:  # e.g. single-parameter study
        logger.warning("Could not compute param importances: %s", exc)
        return {}


def plot_param_importances(optimisation_result: dict):
    """Bar chart of Optuna hyperparameter importances (report Appendix figure)."""
    importances = optimisation_result.get("param_importances", {})
    fig, ax = plt.subplots(figsize=(7, 4))
    if importances:
        names = list(importances.keys())
        vals = [importances[n] for n in names]
        ax.barh(names, vals, color="#2a7f62")
        ax.set_xlabel("Importance (variance in CV ROC-AUC explained)")
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "Importances unavailable", ha="center", va="center")
        ax.axis("off")
    ax.set_title(
        f"Optuna hyperparameter importance — {optimisation_result['champion']} "
        f"({optimisation_result['n_trials']} trials)"
    )
    fig.tight_layout()
    return fig
