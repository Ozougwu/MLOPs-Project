"""data_cleaning pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Node, Pipeline

from .nodes import assemble_model_input


def create_pipeline(**kwargs) -> Pipeline:
    """Re-join feature groups and encode into a clean model-input frame."""
    return Pipeline(
        [
            Node(
                func=assemble_model_input,
                inputs=[
                    "feature_group_numerical",
                    "feature_group_categorical",
                    "feature_group_target",
                ],
                outputs="model_input_table",
                name="assemble_model_input_node",
            ),
        ]
    )
