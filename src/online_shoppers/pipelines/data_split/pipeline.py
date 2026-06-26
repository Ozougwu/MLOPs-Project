"""data_split pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Node, Pipeline

from .nodes import split_train_test


def create_pipeline(**kwargs) -> Pipeline:
    """Stratified train/test split (test sealed until final eval)."""
    return Pipeline(
        [
            Node(
                func=split_train_test,
                inputs=["model_input_table", "params:split"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_train_test_node",
            ),
        ]
    )
