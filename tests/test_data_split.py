"""Tests for the data_split node (req #6).

The split is the backbone of our leakage discipline: the test set must be
sealed (never seen during selection/training), the buyer ratio must be
preserved in both folds (stratification on imbalanced data), and the split must
be reproducible from the seed so every grader gets the same rows.
"""

import numpy as np
import pandas as pd

from online_shoppers.pipelines.data_split.nodes import split_train_test

PARAMS = {"test_fraction": 0.2, "random_state": 42}


def _frame(n: int = 500, buyer_rate: float = 0.15) -> pd.DataFrame:
    """A synthetic model-input frame with an imbalanced integer Revenue target."""
    rng = np.random.default_rng(0)
    y = (rng.random(n) < buyer_rate).astype(int)
    return pd.DataFrame(
        {"PageValues": rng.random(n), "BounceRates": rng.random(n), "Revenue": y}
    )


def test_split_sizes_and_no_overlap():
    """Test fraction is honoured and train/test rows are disjoint (sealed test)."""
    df = _frame()
    X_train, X_test, y_train, y_test = split_train_test(df, PARAMS)
    assert len(X_test) == round(0.2 * len(df))
    assert len(X_train) + len(X_test) == len(df)
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert "Revenue" not in X_train.columns  # target removed from features


def test_stratification_preserves_buyer_ratio():
    """Buyer proportion in train/test stays close to the overall rate."""
    df = _frame()
    overall = df["Revenue"].mean()
    _, _, y_train, y_test = split_train_test(df, PARAMS)
    assert abs(y_train.mean() - overall) < 0.02
    assert abs(y_test.mean() - overall) < 0.03


def test_split_is_reproducible():
    """Same seed -> identical test rows (graders reproduce our results)."""
    df = _frame()
    _, X_test_a, _, _ = split_train_test(df, PARAMS)
    _, X_test_b, _, _ = split_train_test(df, PARAMS)
    assert list(X_test_a.index) == list(X_test_b.index)
