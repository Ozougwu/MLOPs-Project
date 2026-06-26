"""model_selection pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Node, Pipeline

from .nodes import select_champion


def create_pipeline(**kwargs) -> Pipeline:
    """Cross-validate candidates on train; pick the ROC-AUC champion."""
    return Pipeline(
        [
            Node(
                func=select_champion,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:candidates",
                    "params:model_selection",
                ],
                outputs="champion_config",
                name="select_champion_node",
            ),
        ]
    )
