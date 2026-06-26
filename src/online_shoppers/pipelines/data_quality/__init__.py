"""Data quality pipeline (Great Expectations unit data tests + gate)."""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
