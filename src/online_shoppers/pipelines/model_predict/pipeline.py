"""model_predict pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Node, Pipeline

from .nodes import predict_batch


def create_pipeline(**kwargs) -> Pipeline:
    """Batch-score the held-out test features with the champion model."""
    return Pipeline(
        [
            Node(
                func=predict_batch,
                # X_test is the sealed holdout -> an honest "fresh" batch the
                # model never trained on. Swap for any new batch in production.
                inputs=["champion_model", "X_test"],
                outputs="batch_predictions",
                name="predict_batch_node",
            ),
        ]
    )
