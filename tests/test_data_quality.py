"""Tests for the data_quality pipeline."""

import great_expectations as gx
import pandas as pd

from online_shoppers.pipelines.data_quality.nodes import (
    build_expectation_suite,
    validate_raw_data,
)


def _good_row() -> dict:
    return {
        "Administrative": 1,
        "BounceRates": 0.02,
        "ExitRates": 0.04,
        "PageValues": 10.0,
        "Month": "Nov",
        "VisitorType": "Returning_Visitor",
        "Revenue": False,
    }


def test_suite_has_at_least_six_expectations():
    """Brief requires >= 6 data asserts."""
    gx.get_context()  # GX 1.x needs an active context to read suite.expectations
    suite = build_expectation_suite()
    assert len(suite.expectations) >= 6


def test_gate_passes_on_clean_data():
    """A clean batch passes the gate and is returned unchanged."""
    df = pd.DataFrame([_good_row(), _good_row()])
    out = validate_raw_data(df)
    assert len(out) == 2


def test_gate_fails_on_bad_visitor_type():
    """An out-of-set VisitorType must trip the gate (halt the pipeline)."""
    bad = _good_row()
    bad["VisitorType"] = "Robot"
    df = pd.DataFrame([bad])
    try:
        validate_raw_data(df)
    except ValueError:
        return  # expected
    raise AssertionError("Gate should have failed on bad VisitorType")
