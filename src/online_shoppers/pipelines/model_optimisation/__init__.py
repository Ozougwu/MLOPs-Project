"""model_optimisation pipeline (creative extension): Optuna HPO for the champion.

Optional and isolated -- the default pipeline uses the fixed `candidates`
params and never needs this. Running it tunes the champion family's
hyperparameters and emits a `tuned_candidates` config + an Optuna importance
plot, mirroring a real HPO stage without breaking reproducibility.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
