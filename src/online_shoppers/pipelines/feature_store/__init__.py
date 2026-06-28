"""Feature store pipeline (persist engineered features to local parquet)."""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
