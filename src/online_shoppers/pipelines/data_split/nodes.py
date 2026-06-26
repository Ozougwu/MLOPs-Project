"""Nodes for the data_split pipeline.

Splits the model-input frame into TRAIN and TEST only. By design there is no
separate validation set: cross-validation on the training set (model_selection)
provides validation, and the test set is SEALED -- never touched until the final
evaluation in model_train. The split is stratified on the imbalanced target
(~15.5% buyers) and seeded for reproducibility ("same results for everyone").
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

TARGET = "Revenue"


def split_train_test(
    df: pd.DataFrame, params: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split (test sealed for final eval only).

    Args:
        df: model-input frame (features + integer ``Revenue`` target).
        params: ``split`` params -> ``test_fraction``, ``random_state``.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.
    """
    y = df[TARGET]
    X = df.drop(columns=TARGET)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_fraction"],
        random_state=params["random_state"],
        stratify=y,  # preserve the buyer ratio in both splits
    )

    logger.info(
        "Split -> train: %d rows (%.1f%% buyers), test: %d rows (%.1f%% buyers). "
        "Test is SEALED until final eval.",
        len(X_train),
        100 * y_train.mean(),
        len(X_test),
        100 * y_test.mean(),
    )
    return X_train, X_test, y_train, y_test
