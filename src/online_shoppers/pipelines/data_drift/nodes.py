"""data_drift nodes (req #5).

We tell a *natural seasonal drift* story instead of injecting synthetic noise:
buyer behaviour genuinely shifts between low season and the Oct/Nov/Dec holiday
rush, so we compare two real slices of the production-shaped feature table.

  - reference = low-season rows  (is_holiday_season == 0)
  - current   = holiday rows     (is_holiday_season == 1)

Evidently's DataDriftPreset builds the full visual HTML report; we additionally
run a per-column ValueDrift to emit a tidy CSV (column, drift_score, drifted)
for the report appendix and any CI gate. This catches the "silent degradation"
the brief asks about: the model was trained across all months, and if a future
batch drifts far from training, this is how we'd notice before metrics rot.
"""

from __future__ import annotations

import logging

import pandas as pd
from evidently import Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset

logger = logging.getLogger(__name__)

# Columns we do NOT feed to the drift check:
#   - Revenue          : the target/label (drift here is concept drift, reported
#                         separately; the brief's #5 is *data* drift on features)
#   - is_holiday_season: the split key itself -> trivially "drifts" by construction
#   - Month_* / month_num: we split BY month, so these are circular
_EXCLUDE_PREFIXES = ("Month_",)
_EXCLUDE_EXACT = {"Revenue", "is_holiday_season", "month_num"}


def split_reference_current(model_input_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the feature table into a seasonal reference vs current batch.

    Reference = low season (is_holiday_season == 0); current = holiday rush.
    Returns the feature subset only (excludes target + split-defining columns)
    so the drift report reflects genuine behavioural shift, not the split rule.
    """
    if "is_holiday_season" not in model_input_table.columns:
        raise KeyError("model_input_table must carry 'is_holiday_season' for the seasonal split.")

    feature_cols = [
        c
        for c in model_input_table.columns
        if c not in _EXCLUDE_EXACT and not c.startswith(_EXCLUDE_PREFIXES)
    ]

    holiday = model_input_table["is_holiday_season"] == 1
    reference = model_input_table.loc[~holiday, feature_cols].reset_index(drop=True)
    current = model_input_table.loc[holiday, feature_cols].reset_index(drop=True)

    logger.info(
        "Seasonal drift split: reference (low season)=%d rows, current (holiday)=%d rows, %d features.",
        len(reference),
        len(current),
        len(feature_cols),
    )
    if reference.empty or current.empty:
        raise ValueError("One seasonal slice is empty; check is_holiday_season encoding.")
    return reference, current


def build_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> str:
    """Run Evidently's DataDriftPreset and return the rendered HTML report."""
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)
    summary = snapshot.dict()["metrics"][0]["value"]
    logger.info(
        "Dataset drift: %s of %s columns drifted (share=%.2f).",
        summary.get("count"),
        len(reference.columns),
        summary.get("share", 0.0),
    )
    return snapshot.get_html_str(as_iframe=False)


def _parse_method_threshold(metric_name: str) -> tuple[str, float, bool]:
    """Extract method, threshold, and the drift DIRECTION Evidently auto-picked.

    Evidently 0.7 chooses the test per column based on type/cardinality and
    embeds it in the metric name, e.g.:
      - "method=Wasserstein distance (normed),threshold=0.1"  (numeric, many vals)
      - "method=Jensen-Shannon distance,threshold=0.1"        (categorical)
      - "method=K-S p_value,threshold=0.05"                   (numeric, few cols)

    The drift rule flips with the metric family: a *distance* drifts when it
    exceeds the threshold, but a *p-value* drifts when it falls BELOW it. We
    return ``higher_is_drift`` so the caller applies the right comparison and
    the CSV verdict matches the visual DataDriftPreset's own per-column call.
    """
    method = "unknown"
    threshold = 0.1
    if "method=" in metric_name:
        method = metric_name.split("method=", 1)[1].split(",threshold=", 1)[0]
    if "threshold=" in metric_name:
        threshold = float(metric_name.split("threshold=", 1)[1].rstrip(")"))
    # p-value tests reject "no drift" at small values; distances do at large ones.
    higher_is_drift = "p_value" not in method.lower()
    return method, threshold, higher_is_drift


def build_drift_table(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Per-column drift detail as a tidy table for the report/CI gate.

    For each feature Evidently auto-selects a test (Jensen-Shannon / normalised
    Wasserstein distance, or a K-S p-value) and a threshold. A column DRIFTED
    when a distance exceeds the threshold OR a p-value falls below it; we honour
    that direction so the CSV verdict matches the visual DataDriftPreset's own
    per-column call and the dataset-level drift share.
    """
    metrics = [ValueDrift(column=c) for c in reference.columns]
    snapshot = Report(metrics=metrics).run(reference_data=reference, current_data=current)

    rows = []
    for col, m in zip(reference.columns, snapshot.dict()["metrics"]):
        method, threshold, higher_is_drift = _parse_method_threshold(m["metric_name"])
        score = float(m["value"])
        drifted = score > threshold if higher_is_drift else score < threshold
        rows.append(
            {
                "column": col,
                "method": method,
                "drift_score": score,
                "threshold": threshold,
                "drifted": bool(drifted),
            }
        )

    # Sort drifted-first so the most actionable columns lead the report.
    table = pd.DataFrame(rows).sort_values(
        ["drifted", "drift_score"], ascending=[False, False]
    ).reset_index(drop=True)
    n_drift = int(table["drifted"].sum())
    logger.info("Per-column drift: %d / %d features flagged as drifted.", n_drift, len(table))
    return table
