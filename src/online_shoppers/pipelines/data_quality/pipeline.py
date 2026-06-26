"""Pipeline definition for data_quality."""

from __future__ import annotations

from kedro.pipeline import Node, Pipeline

from .nodes import validate_raw_data


def create_pipeline(**kwargs) -> Pipeline:
    """GX gate: validate raw data; output feeds downstream pipelines."""
    return Pipeline(
        [
            Node(
                func=validate_raw_data,
                inputs="online_shoppers_raw",
                outputs="online_shoppers_validated",
                name="validate_raw_data_node",
            ),
        ]
    )
