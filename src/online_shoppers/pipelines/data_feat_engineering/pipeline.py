"""Pipeline definition for data_feat_engineering."""

from __future__ import annotations

from kedro.pipeline import Node, Pipeline

from .nodes import engineer_features


def create_pipeline(**kwargs) -> Pipeline:
    """Bronze -> Silver: take the GX-validated data, add engineered features."""
    return Pipeline(
        [
            Node(
                func=engineer_features,
                inputs="online_shoppers_validated",
                outputs="online_shoppers_featured",
                name="engineer_features_node",
            ),
        ]
    )
