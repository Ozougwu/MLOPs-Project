"""Tests for the model_predict node (req #6).

predict_batch is the batch-inference contract: given the trained Pipeline and a
feature batch, it returns one prediction + probability per row, aligned to the
batch index. We fit a tiny real sklearn Pipeline so the test exercises the same
scaler+classifier shape the champion uses, with no MLflow dependency.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from online_shoppers.pipelines.model_predict.nodes import predict_batch


def _fitted_pipeline() -> tuple[Pipeline, pd.DataFrame]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"PageValues": rng.random(100), "BounceRates": rng.random(100)})
    y = (X["PageValues"] > 0.5).astype(int)  # learnable signal
    pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
    pipe.fit(X, y)
    return pipe, X


def test_output_shape_and_columns():
    """One row out per row in, with the two expected columns."""
    pipe, X = _fitted_pipeline()
    out = predict_batch(pipe, X.iloc[:10])
    assert len(out) == 10
    assert list(out.columns) == ["predicted_revenue", "buyer_probability"]


def test_predictions_are_binary_and_probabilities_in_range():
    pipe, X = _fitted_pipeline()
    out = predict_batch(pipe, X)
    assert set(out["predicted_revenue"].unique()) <= {0, 1}
    assert out["buyer_probability"].between(0.0, 1.0).all()


def test_index_is_preserved():
    """Predictions stay aligned to the input batch index (joinable downstream)."""
    pipe, X = _fitted_pipeline()
    batch = X.iloc[[5, 17, 42]]
    out = predict_batch(pipe, batch)
    assert list(out.index) == [5, 17, 42]
