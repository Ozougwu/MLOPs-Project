"""Tests for the data_cleaning pipeline node (req #6).

assemble_model_input merges the three feature groups, casts bools to int, and
one-hot encodes the categoricals. The contract we protect here: no rows are
lost or duplicated, no nulls leak through, the target survives as 0/1, and the
categorical columns are replaced by their one-hot indicators.
"""

import pandas as pd

from online_shoppers.pipelines.data_cleaning.nodes import assemble_model_input


def _groups() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three tiny feature groups keyed on a shared ``index`` (as in the store)."""
    numerical = pd.DataFrame(
        {"index": [0, 1, 2], "PageValues": [0.0, 12.5, 3.0], "BounceRates": [0.2, 0.0, 0.1]}
    )
    categorical = pd.DataFrame(
        {
            "index": [0, 1, 2],
            "Month": ["Feb", "Nov", "Nov"],
            "VisitorType": ["New_Visitor", "Returning_Visitor", "Returning_Visitor"],
            "Weekend": [True, False, True],
        }
    )
    target = pd.DataFrame({"index": [0, 1, 2], "Revenue": [False, True, False]})
    return numerical, categorical, target


def test_no_rows_lost_and_no_nulls():
    """The merge preserves the row count and introduces no nulls."""
    num, cat, tgt = _groups()
    out = assemble_model_input(num, cat, tgt)
    assert len(out) == 3
    assert not out.isnull().any().any()


def test_target_is_integer_binary():
    """Revenue is cast bool -> {0, 1} so models/metrics treat it numerically."""
    num, cat, tgt = _groups()
    out = assemble_model_input(num, cat, tgt)
    assert set(out["Revenue"].unique()) <= {0, 1}
    assert out["Revenue"].tolist() == [0, 1, 0]


def test_categoricals_one_hot_encoded():
    """Raw Month/VisitorType are replaced by one-hot indicator columns."""
    num, cat, tgt = _groups()
    out = assemble_model_input(num, cat, tgt)
    assert "Month" not in out.columns
    assert "VisitorType" not in out.columns
    assert any(c.startswith("Month_") for c in out.columns)
    # drop_first => one of the two months is the reference and absent as a column
    assert "Weekend" in out.columns
    assert set(out["Weekend"].unique()) <= {0, 1}
