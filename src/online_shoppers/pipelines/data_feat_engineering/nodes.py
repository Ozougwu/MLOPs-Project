"""Nodes for the data_feat_engineering pipeline (Bronze -> Silver).

All transforms here are PER-ROW (no fitting on the data), so they are safe to
run before the train/test split without leakage. Anything that *learns* from
the data (scaler/encoder/imputer) is deferred to data_cleaning, fit on train
only (Sprint 2).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Month string -> calendar number (dataset uses "June", not "Jun").
MONTH_TO_INT = {
    "Feb": 2,
    "Mar": 3,
    "May": 5,
    "June": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add behavioural / temporal features for the Online Shoppers data."""
    data = df.copy()

    # --- Temporal: numeric month + holiday-season flag (used later for drift)
    data["month_num"] = data["Month"].map(MONTH_TO_INT).astype("Int64")
    data["is_holiday_season"] = data["Month"].isin(["Nov", "Dec"]).astype(int)

    # --- Total engagement across the three page categories
    data["total_pages"] = (
        data["Administrative"] + data["Informational"] + data["ProductRelated"]
    )
    data["total_duration"] = (
        data["Administrative_Duration"]
        + data["Informational_Duration"]
        + data["ProductRelated_Duration"]
    )

    # --- Intent signals
    data["has_page_value"] = (data["PageValues"] > 0).astype(int)
    data["avg_duration_per_page"] = data["total_duration"] / data[
        "total_pages"
    ].replace(0, np.nan)
    data["avg_duration_per_page"] = data["avg_duration_per_page"].fillna(0.0)

    # --- Bucket the product-related browsing depth
    data["product_depth_bin"] = pd.cut(
        data["ProductRelated"],
        bins=[-np.inf, 0, 5, 20, 50, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    logger.info(
        "Engineered %d new features; shape now %s.",
        7,
        data.shape,
    )
    return data
