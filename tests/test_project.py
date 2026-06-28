"""Smoke tests that the Kedro project is wired correctly.

Real node/pipeline tests are added per pipeline (Sprint 5, req #6). This file
just proves the test harness, package import, and project metadata are healthy
so `pytest` is green from Sprint 0 onward.
"""

from pathlib import Path

import online_shoppers


def test_package_imports():
    """The project package imports and exposes a version."""
    assert online_shoppers.__version__


def test_raw_data_present():
    """The raw dataset is in the expected Kedro layer."""
    raw = Path(__file__).parent.parent / "data" / "01_raw" / "online_shoppers.csv"
    assert raw.exists(), f"Expected raw data at {raw}"
