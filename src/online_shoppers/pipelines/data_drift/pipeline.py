"""data_drift pipeline definition (req #5).

Runs independently of model training: it consumes the persisted feature table,
so a grader can `kedro run --pipeline=data_drift` on its own.
"""

from __future__ import annotations

from kedro.pipeline import Node, Pipeline

from .nodes import build_drift_report, build_drift_table, split_reference_current


def create_pipeline(**kwargs) -> Pipeline:
    """Seasonal reference vs current -> Evidently HTML report + drift CSV."""
    return Pipeline(
        [
            Node(
                func=split_reference_current,
                inputs="model_input_table",
                outputs=["drift_reference", "drift_current"],
                name="split_reference_current_node",
            ),
            Node(
                func=build_drift_report,
                inputs=["drift_reference", "drift_current"],
                outputs="data_drift_report",
                name="build_drift_report_node",
            ),
            Node(
                func=build_drift_table,
                inputs=["drift_reference", "drift_current"],
                outputs="drift_result",
                name="build_drift_table_node",
            ),
        ]
    )
