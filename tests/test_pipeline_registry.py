"""Orchestration / pipeline tests (req #6).

The brief requires modular pipelines that run *separately* and a working
__default__ chain. These tests assert the registry exposes every named
pipeline, that each is an independently constructable Pipeline object, and that
a representative pipeline (data_drift) wires its nodes into a runnable DAG. This
is the "at least one pipeline test" plus the orchestration contract.
"""

from kedro.pipeline import Pipeline

from online_shoppers.pipeline_registry import register_pipelines

# The nine REQUIRED modular pipelines the project is built from (brief's diagram).
REQUIRED_PIPELINES = {
    "data_quality",
    "feature_store",
    "data_cleaning",
    "data_feat_engineering",
    "data_split",
    "model_selection",
    "model_train",
    "model_predict",
    "data_drift",
}

# Optional creative extension (Optuna HPO); isolated, not in the default chain.
STRETCH_PIPELINES = {"model_optimisation"}

EXPECTED_PIPELINES = REQUIRED_PIPELINES | STRETCH_PIPELINES


def test_all_nine_required_pipelines_registered():
    """Every required modular pipeline is discoverable and runnable on its own."""
    pipelines = register_pipelines()
    missing = REQUIRED_PIPELINES - set(pipelines)
    assert not missing, f"Missing pipelines: {missing}"


def test_stretch_pipeline_registered():
    """The Optuna HPO extension is registered as an independent pipeline."""
    pipelines = register_pipelines()
    assert STRETCH_PIPELINES <= set(pipelines)


def test_default_pipeline_is_the_full_chain():
    """__default__ exists and aggregates the named pipelines (full kedro run)."""
    pipelines = register_pipelines()
    assert "__default__" in pipelines
    assert isinstance(pipelines["__default__"], Pipeline)
    assert len(pipelines["__default__"].nodes) > 0


def test_each_pipeline_is_independently_constructable():
    """Each named pipeline is a Pipeline with at least one node (separately runnable)."""
    pipelines = register_pipelines()
    for name in EXPECTED_PIPELINES:
        pipe = pipelines[name]
        assert isinstance(pipe, Pipeline), f"{name} is not a Pipeline"
        assert len(pipe.nodes) >= 1, f"{name} has no nodes"


def test_data_drift_pipeline_dag_is_wired():
    """data_drift links split -> report/table via the shared intermediate frames."""
    drift = register_pipelines()["data_drift"]
    node_names = {n.name for n in drift.nodes}
    assert {
        "split_reference_current_node",
        "build_drift_report_node",
        "build_drift_table_node",
    } <= node_names
    # the split feeds both downstream nodes -> its outputs are the report inputs
    assert {"drift_reference", "drift_current"} <= set(drift.all_outputs())
