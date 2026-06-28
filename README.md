# Online Shoppers Purchasing Intention — MLOps Pipeline

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

MSc MLOps capstone. An end-to-end, modular, reproducible Kedro pipeline that
predicts whether an e-commerce visitor will make a purchase (`Revenue`), using
the **Online Shoppers Purchasing Intention** dataset (UCI, ~12,330 rows,
~15.5 % buyers — imbalanced, so we report **recall on buyers / F1 / ROC-AUC**,
not accuracy).

## Pipelines (each runs independently or as the full chain)

| Pipeline | Requirement | What it does |
|---|---|---|
| `data_quality` | #1 | Great Expectations gate (8 asserts); halts on bad data |
| `data_feat_engineering` | — | Per-row behavioural/temporal features (no leakage) |
| `feature_store` | #1 | Persists feature groups to local parquet (`data/04_feature/`) |
| `data_cleaning` | — | Encode/clean (learned transforms fit on train only) |
| `data_split` | — | Stratified train/test; test held out until final eval |
| `model_selection` | — | Cross-validated champion selection (ROC-AUC) |
| `model_train` | #2, #3 | Train + MLflow tracking/registry + SHAP explainability |
| `model_predict` | — | Batch inference on the sealed test set (champion model) |
| `data_drift` | #5 | Evidently drift report (natural seasonal drift) |

## Quickstart (reproducible — uses the committed sample data)

A 1,000-row stratified **sample** (`data/01_raw/online_shoppers_sample.csv`,
same 15.5 % buyer ratio) is committed so the pipeline runs out of the box. The
full dataset is gitignored; see `helpers/data_download.md` to fetch it.

```bash
# 1. Install dependencies (uv shown; plain `pip install -r requirements.txt` also works)
uv pip install -r requirements.txt

# 2. (First time only) initialise MLflow config — a committed conf/base/mlflow.yml
#    already provides a working default, so this is optional.
kedro mlflow init

# 3. Run the full pipeline (data_quality -> ... -> model_train -> data_drift)
kedro run

# 4. Run a single pipeline in isolation (e.g. just the data quality gate)
kedro run --pipeline=data_quality
kedro run --pipeline=data_drift

# 5. Inspect experiments / models
mlflow ui            # http://127.0.0.1:5000
```

To run against the full dataset, point the `online_shoppers_raw` catalog entry
at `online_shoppers.csv` instead of the sample (see `conf/base/catalog.yml`).

## Model serving (#4)

A FastAPI app loads the portable champion model (no absolute training paths, so
it runs identically in a clone or a container) and serves predictions.

```bash
# Build and run the container (loads data/06_models/champion_model + schema)
docker build -f serving/Dockerfile -t online-shoppers-serving .
docker run -p 8000:8000 online-shoppers-serving

# Probe it
curl localhost:8000/health           # liveness
curl localhost:8000/ready            # readiness + expected feature schema
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"rows": [{"PageValues": 12.0, "ProductRelated": 30}]}'
```

`GET /ready` returns the exact feature columns the model expects; missing
one-hot columns default to 0, unknown columns return 422.

## Tests (#6)

```bash
pytest          # node + pipeline/orchestration tests with coverage
ruff check .    # lint
```

Tests cover the cleaning/split/predict/drift nodes plus a registry test that
asserts all nine pipelines are registered and independently runnable.

## Reproducibility notes

* MLflow config is committed at `conf/base/mlflow.yml` (no secrets); any local
  overrides/credentials stay in `conf/local/` (gitignored).
* Every dataset goes through the Kedro catalog — no hardcoded paths in nodes.
* Package versions are pinned in `requirements.txt`.
* **Verified from a clean virtual environment:** installing only the pinned
  `requirements.txt` (plus `pip install -e .`) reproduces the pipeline results
  exactly (e.g. `model_predict` scores 2466 rows → 456 predicted buyers).
* **Windows note:** `shap` pulls in `numba`, whose DLL can fail to load under a
  very deep install path (Windows `MAX_PATH` limit). If you hit
  `ImportError: DLL load failed while importing _box`, clone to a short path
  (e.g. `C:\mlops`) or enable Windows long-path support.


## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. You can install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, 'session', `catalog`, and `pipelines`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/deploy/package_a_project/#package-an-entire-kedro-project)
