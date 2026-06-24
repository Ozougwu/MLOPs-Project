# CLAUDE.md — Online Shoppers MLOps Project

Guidance for Claude Code when working in this repo. Read this first; it prevents re-discovery and reduces wasted tool calls.

## TL;DR for the agent
- This is an **MSc MLOps capstone** (graded). Goal: a Kedro pipeline covering data tests, feature store, MLflow, SHAP, serving, and drift. Target = 20/20.
- **Dataset is LOCKED: Online Shoppers Purchasing Intention** (`data/01_raw/online_shoppers.csv`, 12,330 rows, target `Revenue`). Do not re-litigate the dataset choice.
- **Always use the existing venv** (below). Never create a new env, never `pip install` globally, never run `kedro new` again.
- Plans live one level up in `../` (`PROJECT_PLAN_MERGED.md`, `implementation_plan.md`, `tasks.md`, `data_download.md`, `notebooks/01_eda.ipynb`). The merged plan + tasks are the source of truth for scope and sprint order.

## Environment (CRITICAL — memorize, don't re-detect)
- **Python venv (uv-managed):** `C:\Users\USER\my_mlops_project\.venv-mlops`
- **Python exe:** `/c/Users/USER/my_mlops_project/.venv-mlops/Scripts/python` (Git Bash path)
- This venv is **uv-managed and has NO pip**. To install packages, use uv from the venv's parent:
  ```bash
  cd "/c/Users/USER/my_mlops_project" && VIRTUAL_ENV="/c/Users/USER/my_mlops_project/.venv-mlops" uv pip install <pkg>
  ```
- **Do not** try `python -m pip` inside the venv — it fails (no pip module).
- There is also a separate Anaconda (`C:\Users\USER\anaconda3`) with Jupyter/seaborn — used **only** for running notebooks (e.g. EDA). The pipeline always uses `.venv-mlops`.

### Installed in .venv-mlops (verified — don't reinstall)
kedro 1.3.1 · kedro-datasets[pandas] 9.4.0 · kedro-mlflow 2.0.2 · mlflow 3.11.1 · great_expectations 1.16.1 · shap 0.52 · fastapi 0.136 · uvicorn 0.44 · evidently 0.7.21 · scikit-learn 1.8 · pandas 2.3.3 · optuna 4.8 · pytest 9.1.1
- **Missing on purpose:** `featuretools` (only needed for the optional DFS stretch — install only if that sprint is reached).

## Running things (canonical commands)
Run all from the project root `online-shoppers-mlops/`. Prefix python with the venv path.
```bash
PY="/c/Users/USER/my_mlops_project/.venv-mlops/Scripts/python"
# load/inspect project
$PY -m kedro registry list
# run full pipeline / one pipeline
$PY -m kedro run
$PY -m kedro run --pipeline=data_quality
# tests
$PY -m pytest -q
# mlflow ui (when needed)
$PY -m mlflow ui
```
- Telemetry is disabled via `.telemetry` (consent: false) — leave it.

## Repo layout
```
online-shoppers-mlops/            <- project root (run commands here)
├── CLAUDE.md                     <- this file
├── conf/base/{catalog.yml, parameters.yml}
├── conf/local/                   <- gitignored secrets; mlflow.yml goes here
├── data/01_raw..08_reporting/    <- Kedro data layers; raw CSV already placed
├── src/online_shoppers/
│   ├── pipelines/<name>/{nodes.py, pipeline.py}   <- one folder per pipeline
│   ├── pipeline_registry.py      <- register named + __default__ pipelines
│   └── settings.py
├── tests/                        <- pytest (req #6)
└── serving/                      <- FastAPI + Dockerfile (req #4, created in Sprint 3)
```

## Conventions for this project
- **Pipelines to build (sprint order):** data_quality → data_feat_engineering → feature_store → data_cleaning → data_split → model_selection → model_train → model_predict → data_drift. Each is a Kedro pipeline registered by name AND included in `__default__`.
- **Reproducibility is graded.** Every dataset goes through the catalog (no hardcoded paths in nodes). Pin versions. The grader must be able to `kedro run` and reproduce results.
- **Leakage rule:** per-row feature engineering can happen before split; anything that *learns* (scaler/imputer/encoder/selection) is fit on **train only** and saved as an artifact (avoids train-serve skew).
- **Metric:** target is imbalanced (~15.5% buy) → report **recall on buyers / F1 / ROC-AUC**, never accuracy.
- **Drift (req #5):** use the **natural seasonal drift** — reference = lower-season months, current = holiday months (Nov). Don't fabricate drift by random perturbation.
- **Feature store:** default to a **local parquet store** in `data/04_feature/` (NOT cloud Hopsworks) so the project is reproducible offline. Hopsworks path is documented only.
- **DQ approach (Sprint 1):** Great Expectations, mirroring `Theory Classes/week_01/01_Data_Unit_Tests.ipynb` (ephemeral context → ExpectationSuite → ValidationDefinition → gate node). ≥6 asserts derived from the EDA.

## Reference material (read-only; in ../../Theory Classes/ and ../../Practical/)
- GX pattern: `Practical/week_01/01_Data_Unit_Tests.ipynb`
- Feature store/utils: `Theory Classes/week_02/feature_utils.py`
- Kedro + MLflow + SHAP reference impl: `Theory Classes/week_05/bank-example/`
- Serving reference: `Theory Classes/week_05/spotify_recommender/`
- EDA already done: `../notebooks/01_eda.ipynb` (findings mapped to sprints)

## Token / permission efficiency rules for the agent
- **Don't re-run discovery** that this file already answers (env location, installed pkgs, dataset, layout). Trust it; verify only if a command fails.
- **Batch independent shell calls** into one block; don't sequentially probe.
- **Don't reinstall** packages listed above. Check imports only if an error occurs.
- **Don't re-read large notebooks/PDFs** already summarized in `../` plans unless a specific detail is needed; prefer the plan files.
- Prefer `kedro registry list` / a tiny `$PY -c "..."` over spinning up full runs to sanity-check.
- Keep edits surgical (Edit over full rewrites) once files exist.
- When a step is verified, state it once and move on — no redundant re-checking.
