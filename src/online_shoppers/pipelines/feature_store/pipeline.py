"""Pipeline definition for feature_store."""

from __future__ import annotations

from kedro.pipeline import Node, Pipeline

from .nodes import split_feature_groups


def create_pipeline(**kwargs) -> Pipeline:
    """Persist engineered features as local-parquet feature groups."""
    return Pipeline(
        [
            Node(
                func=split_feature_groups,
                inputs="online_shoppers_featured",
                outputs=[
                    "feature_group_numerical",
                    "feature_group_categorical",
                    "feature_group_target",
                ],
                name="split_feature_groups_node",
            ),
        ]
    )
