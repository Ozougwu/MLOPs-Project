"""Nodes for the data_quality pipeline.

Mirrors the class GX pattern (``Practical/week_01/01_Data_Unit_Tests.ipynb``):
ephemeral context -> ExpectationSuite -> ValidationDefinition -> gate node.
Expectations are derived from the EDA of the raw Online Shoppers data, not
guessed: see the value sets / ranges below.
"""

from __future__ import annotations

import logging

import great_expectations as gx
import pandas as pd

logger = logging.getLogger(__name__)

# Valid value sets / ranges taken directly from the raw-data profile.
VALID_MONTHS = ["Aug", "Dec", "Feb", "Jul", "June", "Mar", "May", "Nov", "Oct", "Sep"]
VALID_VISITOR_TYPES = ["New_Visitor", "Returning_Visitor", "Other"]


def build_expectation_suite() -> gx.ExpectationSuite:
    """Build the Online Shoppers expectation suite (>= 6 asserts).

    Each expectation encodes a data contract learned from the EDA. If any of
    these break on a future batch, the gate node halts the pipeline.
    """
    suite = gx.ExpectationSuite(name="online_shoppers_quality_v1")

    expectations = [
        # 1. Target must only ever be the two boolean strings/values.
        gx.expectations.ExpectColumnDistinctValuesToBeInSet(
            column="Revenue", value_set=[True, False, "True", "False"]
        ),
        # 2. Target must never be missing (no label = useless row).
        gx.expectations.ExpectColumnValuesToNotBeNull(column="Revenue"),
        # 3. VisitorType is a closed category set.
        gx.expectations.ExpectColumnDistinctValuesToBeInSet(
            column="VisitorType", value_set=VALID_VISITOR_TYPES
        ),
        # 4. Month is a closed set (note the dataset uses "June", not "Jun").
        gx.expectations.ExpectColumnDistinctValuesToBeInSet(
            column="Month", value_set=VALID_MONTHS
        ),
        # 5. BounceRates is a proportion in [0, 1].
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="BounceRates", min_value=0.0, max_value=1.0
        ),
        # 6. ExitRates is a proportion in [0, 1].
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="ExitRates", min_value=0.0, max_value=1.0
        ),
        # 7. PageValues is a non-negative monetary value.
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="PageValues", min_value=0.0, max_value=10000.0
        ),
        # 8. Administrative is a non-negative page count.
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="Administrative", min_value=0, max_value=None
        ),
    ]
    for expectation in expectations:
        suite.add_expectation(expectation)

    logger.info("Built suite with %d expectations.", len(expectations))
    return suite


def validate_raw_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Run the suite against the raw data and gate the pipeline.

    Returns the validated DataFrame so downstream pipelines depend on this node
    (the gate). Raises if validation fails, halting ``kedro run``.
    """
    context = gx.get_context()  # ephemeral, in-memory context

    data_source = context.data_sources.add_pandas(name="online_shoppers_source")
    data_asset = data_source.add_dataframe_asset(name="raw_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("raw_batch")

    suite = build_expectation_suite()
    context.suites.add(suite)

    validation_definition = context.validation_definitions.add(
        gx.core.validation_definition.ValidationDefinition(
            name="raw_validation",
            data=batch_definition,
            suite=suite,
        )
    )

    results = validation_definition.run(batch_parameters={"dataframe": df_raw})

    n_total = len(results.results)
    n_passed = sum(1 for r in results.results if r.success)
    logger.info("GX validation: %d/%d expectations passed.", n_passed, n_total)

    if not results.success:
        failed = [r.expectation_config.type for r in results.results if not r.success]
        raise ValueError(
            f"Data quality gate FAILED. {n_total - n_passed} expectation(s) "
            f"failed: {failed}. Halting pipeline."
        )

    logger.info("Data quality gate PASSED. Proceeding.")
    return df_raw
