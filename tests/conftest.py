"""Shared pytest fixtures.

Kedro's ``find_pipelines`` (used by the registry) requires the project to be
bootstrapped first, otherwise ``PACKAGE_NAME`` is unset. We bootstrap once per
test session so the registry/orchestration tests can import the project's
pipelines without spinning up a full KedroSession.
"""

from pathlib import Path

import pytest
from kedro.framework.startup import bootstrap_project

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session", autouse=True)
def bootstrap_kedro_project():
    """Configure the Kedro project so find_pipelines() works in tests."""
    bootstrap_project(PROJECT_ROOT)
