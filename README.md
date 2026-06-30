# Online Shoppers Purchasing Intention — MLOps Pipeline

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

MLOps Project

---

## What we built and why

E-commerce conversion is highly imbalanced: only **15.5% of sessions** in this dataset result in a purchase. Predicting that minority class matters enormously — a missed buyer is a lost sale; a false alarm wastes marketing spend. We built a **production-grade, end-to-end MLOps system** around this problem using the [Online Shoppers Purchasing Intention dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (UCI, 12,330 sessions, 18 features, binary target `Revenue`).

Because the class is imbalanced we report **ROC-AUC, F1, and recall on the buyer class** throughout — not accuracy, which would be misleading.

### Results (champion: Random Forest)

| Metric | Value |
|---|---|
| CV ROC-AUC (train, 5-fold) | 0.9316 |
| Test ROC-AUC (held-out 20%) | 0.927 |
| Test recall — buyer class | 0.730 |
| Test F1 — buyer class | 0.666 |
| Test precision — buyer class | 0.612 |

Optuna TPE hyperparameter optimisation (creative extension) further lifted CV ROC-AUC to **0.9347** (n_estimators=400, max_depth=14).

---

## System architecture

The project is structured as **9 modular Kedro pipelines**, each runnable independently or as the full chain via `kedro run`. Every dataset flows through the Kedro catalog — no hardcoded paths inside nodes.

```
Raw CSV
  └─▶ data_quality        Great Expectations gate (8 assertions)
  └─▶ data_feat_engineering  Per-row behavioural & temporal features (no leakage)
  └─▶ feature_store       Persist feature groups to local Parquet store
  └─▶ data_cleaning       Encode categoricals (fit on train only)
  └─▶ data_split          Stratified 80/20 train-test split
  └─▶ model_selection     5-fold CV across 3 candidates; champion selected by ROC-AUC
  └─▶ model_train         Final fit + MLflow tracking/registry + SHAP explainability
  └─▶ model_predict       Batch inference on the sealed test set
  └─▶ data_drift          Evidently drift report (seasonal reference vs. holiday batch)

+ model_optimisation      Optuna TPE HPO (creative extension, isolated, optional)
```

### Pipeline breakdown

| Pipeline | Brief req | What it does |
|---|---|---|
| `data_quality` | #1 | Great Expectations suite: 8 assertions (null checks, value ranges, cardinality, buyer-rate bounds). Pipeline halts if any assertion fails. |
| `data_feat_engineering` | — | Engineers per-row features before the split: `high_value_page`, `engagement_score`, `is_weekend_shopper`, `bounce_exit_ratio`, `quarter`. No global statistics used — no leakage. |
| `feature_store` | #1 | Splits engineered features into **3 Parquet feature groups** (numerical, categorical, target) stored in `data/04_feature/`. Mirrors Hopsworks feature-group patterns but runs offline for grader reproducibility. |
| `data_cleaning` | — | One-hot encodes `Month`, `VisitorType`, `Weekend`. Encoders are fit on **train only** and saved as Kedro artifacts to prevent train-serve skew. |
| `data_split` | — | Stratified train/test split (80/20, seed=42). Test set is sealed and never seen until final evaluation in `model_train`. |
| `model_selection` | — | 5-fold stratified CV across Logistic Regression, Random Forest, Gradient Boosting (all with `class_weight=balanced`). Selects champion by ROC-AUC. |
| `model_train` | #2, #3 | Fits a `sklearn.Pipeline(StandardScaler + champion)` on train only. MLflow `autolog` captures params and model signature; test metrics and SHAP summary plot are logged manually. Model registered to MLflow Model Registry. |
| `model_predict` | — | Loads the registered champion and runs batch inference on the sealed test set. Scores 2,466 rows → 456 predicted buyers. |
| `data_drift` | #5 | Evidently `DataDriftPreset`: reference = low-season months, current = November holiday traffic. Detected drift in 10/22 columns (share 0.455), below the 0.5 dataset-level threshold — no alarm, but per-column drift flagged for monitoring. |

---

## Key technical decisions

**Feature store — local Parquet, not cloud Hopsworks.**
We mirror the Hopsworks feature-group pattern (numerical / categorical / target groups with an index key for re-joining) but persist to `data/04_feature/` as Parquet files. This makes the project fully reproducible offline without API keys — a grader can `kedro run` from a clean clone with no external dependencies.

**Leakage prevention.**
Everything that *learns* (StandardScaler, one-hot encoders) is fit exclusively on the training split and saved as a Kedro catalog artifact. The feature engineering step uses only per-row arithmetic — no global statistics — so it can safely run before the split.

**Imbalance handling.**
We use `class_weight="balanced"` on all candidates instead of synthetic oversampling (SMOTE). This avoids introducing synthetic buyers into evaluation while still penalising misclassification of the minority class more heavily.

**Serving — portable model directory.**
The FastAPI app loads the champion as a self-contained MLflow model directory (`data/06_models/champion_model/`) — no absolute training paths, works identically in a container or any fresh clone. The model is a bundled `sklearn.Pipeline(scaler + model)` so the API never needs to re-scale inputs itself.

**SHAP explainability.**
`TreeExplainer` is used for Random Forest / Gradient Boosting (exact and fast), `LinearExplainer` for Logistic Regression. SHAP runs on the scaled feature space to match what the model actually sees. The summary plot is logged as an MLflow artifact and saved to `data/08_reporting/`.

**Drift — natural seasonality.**
Instead of fabricating drift by random perturbation, we use the dataset's genuine seasonal signal: reference = Feb–Sep (low season), current = November (holiday peak). This is ecologically valid and demonstrates that the monitoring system detects real-world distribution shift.

---

## Technology stack

| Tool | Role |
|---|---|
| **Kedro 1.3** | Modular pipeline orchestration, catalog, node wiring |
| **MLflow 3.11** | Experiment tracking, model registry, artifact store |
| **Great Expectations 1.16** | Data quality gate (8 assertions, ephemeral context) |
| **SHAP 0.52** | Model explainability (TreeExplainer / LinearExplainer) |
| **Evidently 0.7** | Data drift detection (`DataDriftPreset`, Wasserstein distance) |
| **scikit-learn 1.8** | Models (RF, GB, LR), preprocessing, CV |
| **FastAPI 0.136 + Uvicorn** | REST prediction API (`/predict`, `/health`, `/ready`) |
| **Docker** | Container image for the serving layer |
| **Kubernetes + KIND + HPA** | Auto-scaling deployment (creative extension; 2→6 replicas at 60% CPU) |
| **Optuna 4.8** | Bayesian HPO via TPE sampler (creative extension) |
| **Parquet (via pandas)** | Feature store persistence format |
| **pytest 9.1** | 24 tests covering nodes, pipelines, and registry |
| **kedro-mlflow 2.0** | MLflow integration inside Kedro runs |

---

## Quickstart

A 1,000-row stratified sample (`data/01_raw/online_shoppers_sample.csv`, same 15.5% buyer ratio) is committed so the pipeline runs out of the box without downloading anything.

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -e .

# 2. Run the full pipeline
kedro run

# 3. Run a single pipeline in isolation
kedro run --pipeline=data_quality
kedro run --pipeline=data_drift
kedro run --pipeline=model_optimisation   # Optuna HPO (optional)

# 4. Inspect experiments
mlflow ui    # open http://127.0.0.1:5000

# 5. Run tests
pytest
```

> **Full dataset:** swap `online_shoppers_raw` in `conf/base/catalog.yml` to point at `online_shoppers.csv` (see download instructions in the project report).

---

## Model serving

```bash
# Build and run the container
docker build -f serving/Dockerfile -t online-shoppers-serving .
docker run -p 8000:8000 online-shoppers-serving

# Liveness + readiness probes
curl localhost:8000/health
curl localhost:8000/ready          # also returns the 33 expected feature columns

# Predict (send already-engineered feature rows)
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"rows": [{"PageValues": 12.0, "ProductRelated": 30}]}'
```

`GET /ready` returns the exact 33-column feature schema the model expects. Missing one-hot columns default to 0; unknown columns return HTTP 422.

---

## Kubernetes (creative extension)

Manifests in `serving/k8s/`:

```bash
kubectl apply -f serving/k8s/deployment.yaml
kubectl apply -f serving/k8s/service.yaml
kubectl apply -f serving/k8s/hpa.yaml     # HPA: 2→6 replicas at 60% CPU
```

NodePort exposed on port 30080. HPA configured with `minReplicas=2`, `maxReplicas=6`, `targetCPUUtilizationPercentage=60`.

---

## Project structure

```
.
├── conf/base/
│   ├── catalog.yml          # all datasets (01_raw → 08_reporting)
│   └── parameters.yml       # model/split/Optuna settings
├── data/
│   ├── 01_raw/              # committed sample CSV
│   ├── 04_feature/          # local Parquet feature store (generated)
│   ├── 06_models/           # champion model (MLflow format)
│   └── 08_reporting/        # SHAP plot, drift HTML report, metrics
├── docs/                    # project report + figures
├── notebooks/01_eda.ipynb   # exploratory analysis
├── serving/
│   ├── app/main.py          # FastAPI app
│   ├── Dockerfile
│   └── k8s/                 # Kubernetes manifests
├── src/online_shoppers/
│   ├── pipelines/<name>/    # nodes.py + pipeline.py per pipeline
│   ├── pipeline_registry.py
│   └── settings.py
├── tests/                   # 24 pytest tests
├── pyproject.toml
└── requirements.txt
```

---

## Reproducibility

- All datasets go through the Kedro catalog — no hardcoded paths in any node.
- `conf/base/mlflow.yml` is committed; `conf/local/` (secrets/overrides) is gitignored.
- Package versions are pinned in `requirements.txt`.
- Verified from a clean virtual environment: `kedro run` reproduces all results exactly (`model_predict` scores 2,466 rows → 456 predicted buyers).
- **Windows note:** `shap` pulls in `numba`, whose DLL can fail on deep install paths (Windows `MAX_PATH` limit). If you see `ImportError: DLL load failed while importing _box`, clone to a short path (e.g. `C:\mlops`) or enable long-path support in Windows settings.
