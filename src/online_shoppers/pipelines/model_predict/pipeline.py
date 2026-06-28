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
                # Load the PORTABLE local champion (data/06_models/champion_model),
                # not the registry dataset: the registry version needs the
                # training run's context to resolve, so it can't load in a
                # standalone `kedro run --pipeline=model_predict`. The local dir
                # has no absolute paths -> this pipeline runs independently and
                # reproducibly (same model the serving container loads).
                #
                # X_test is the sealed holdout -> an honest "fresh" batch the
                # model never trained on. Swap for any new batch in production.
                inputs=["champion_model_local", "X_test"],
                outputs="batch_predictions",
                name="predict_batch_node",
            ),
        ]
    )
