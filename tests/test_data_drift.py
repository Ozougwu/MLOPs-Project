"""Tests for the data_drift nodes (req #5 + #6).

We protect two contracts: (1) the seasonal split excludes the target and the
split-defining columns so the drift report reflects genuine behaviour, not the
split rule; (2) the per-column table flags drift when the distance exceeds
Evidently's auto-selected threshold, agreeing with the visual preset.
"""

import numpy as np
import pandas as pd

from online_shoppers.pipelines.data_drift.nodes import (
    build_drift_table,
    split_reference_current,
)


def _input_table(n: int = 400) -> pd.DataFrame:
    """A model-input-shaped frame with a seasonal split key and a clear shift."""
    rng = np.random.default_rng(0)
    holiday = np.array([0] * (n // 2) + [1] * (n // 2))
    # PageValues shifts hard between seasons; BounceRates barely moves.
    page = np.where(holiday == 1, rng.normal(20, 2, n), rng.normal(2, 2, n))
    return pd.DataFrame(
        {
            "PageValues": page,
            "BounceRates": rng.normal(0.1, 0.01, n),
            "is_holiday_season": holiday,
            "month_num": np.where(holiday == 1, 11, 3),
            "Month_Nov": holiday,
            "Revenue": (rng.random(n) < 0.15).astype(int),
        }
    )


def test_split_excludes_target_and_split_keys():
    """Reference/current carry only genuine features, not the split machinery."""
    ref, cur = split_reference_current(_input_table())
    for excluded in ("Revenue", "is_holiday_season", "month_num", "Month_Nov"):
        assert excluded not in ref.columns
        assert excluded not in cur.columns
    assert not ref.empty and not cur.empty
    assert list(ref.columns) == list(cur.columns)


def test_drift_table_flags_the_shifted_feature():
    """PageValues (shifted hard by construction) must be flagged as drifted.

    We assert on the ``drifted`` verdict rather than the raw score, because
    Evidently auto-selects the test (distance vs p-value) by column count, and
    the two families move in opposite directions -- the node's drift flag
    normalises that, which is exactly the behaviour we want to protect.
    """
    ref, cur = split_reference_current(_input_table())
    table = build_drift_table(ref, cur)
    by_col = table.set_index("column")
    assert bool(by_col.loc["PageValues", "drifted"]) is True
    # required columns are present and the flag is a clean bool
    assert "BounceRates" in by_col.index
    assert by_col.loc["PageValues", "method"]  # method label populated
