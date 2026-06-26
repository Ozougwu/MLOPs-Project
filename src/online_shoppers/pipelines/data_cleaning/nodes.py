"""Nodes for the data_cleaning pipeline (feature groups -> model-input frame).

Re-joins the three local-parquet feature groups on their ``index`` key, then
applies STRUCTURAL, per-column transforms only:
  - one-hot encode the string categoricals (Month, VisitorType),
  - cast the boolean Weekend / Revenue to integers.

These are leakage-safe: the encoded column set is fixed by the validated schema
(data_quality already asserts the allowed Month / VisitorType values), not
learned from row values. The only transform that *learns* from data -- the
StandardScaler -- is deliberately deferred to model_train and fit on TRAIN only.

We intentionally do NOT de-duplicate: the EDA found ~125 identical rows that are
legitimate repeat sessions, not data errors.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

TARGET = "Revenue"
# String categoricals that need one-hot encoding.
ONE_HOT_COLS = ["Month", "VisitorType"]
# Boolean-like columns cast straight to int (no expansion).
BOOL_COLS = ["Weekend"]


def assemble_model_input(
    numerical: pd.DataFrame,
    categorical: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    """Re-join the feature groups and encode categoricals into a clean frame.

    Args:
        numerical: numerical feature group (carries ``index``).
        categorical: categorical feature group (carries ``index``).
        target: target group with ``index`` + ``Revenue``.

    Returns:
        One model-input frame: numeric features + one-hot categoricals + the
        integer target ``Revenue``, with the join key dropped.
    """
    # --- Re-join on the feature-store key, then drop it.
    df = numerical.merge(categorical, on="index", how="inner").merge(
        target, on="index", how="inner"
    )
    n_before = len(df)
    df = df.drop(columns="index")

    # --- Boolean -> int (Weekend, and the target).
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)
    df[TARGET] = df[TARGET].astype(int)

    # --- One-hot encode string categoricals (drop_first avoids collinearity).
    present = [c for c in ONE_HOT_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=present, drop_first=True)

    # Ensure dummy columns are int (not bool) for downstream estimators.
    dummy_cols = [
        c for c in df.columns if any(c.startswith(f"{p}_") for p in present)
    ]
    df[dummy_cols] = df[dummy_cols].astype(int)

    assert len(df) == n_before, "Row count changed during cleaning (join issue)."
    assert df.isnull().sum().sum() == 0, "Nulls present after cleaning."

    logger.info(
        "Model-input assembled: %d rows x %d cols (target=%s, one-hot from %s).",
        df.shape[0],
        df.shape[1],
        TARGET,
        present,
    )
    return df
