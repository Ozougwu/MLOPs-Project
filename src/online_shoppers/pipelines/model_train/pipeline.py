"""model_train pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Node, Pipeline

from .nodes import (
    explain_champion,
    export_feature_columns,
    export_portable_model,
    train_champion,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Fit champion (scaler train-only), eval on sealed test, SHAP + MLflow."""
    return Pipeline(
        [
            Node(
                func=train_champion,
                inputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                    "champion_config",
                    "params:candidates",
                    "params:model_selection",
                ],
                outputs=["champion_model", "champion_metrics"],
                name="train_champion_node",
            ),
            Node(
                func=explain_champion,
                inputs=["champion_model", "X_train"],
                outputs="shap_summary_plot",
                name="explain_champion_node",
            ),
            Node(
                func=export_portable_model,
                inputs="champion_model",
                outputs="champion_model_local",
                name="export_portable_model_node",
            ),
            Node(
                func=export_feature_columns,
                inputs="X_train",
                outputs="feature_columns",
                name="export_feature_columns_node",
            ),
        ]
    )
