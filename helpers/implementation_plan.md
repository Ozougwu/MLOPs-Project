# Implementation Plan — MSc MLOps Project (20/20 Delivery)

> Planned via the **Andru.ia Solutions Architect** skill. Per user override, all communication and artifacts are in **English** (the skill defaults to Spanish).

---

## 🤖 Andru.ia Diagnosis & Technical Prescription

**Scenario B — Evolution Project.** The workspace contains preexisting course material (6 weeks of theory, reference Kedro projects in `week_05`, practical notebooks). `Practical/` and `Main Project/` are empty: we build the project from scratch by **evolving** the patterns already taught.

**Technical Prescription:**
- **Available stack (all demonstrated in `Theory Classes/week_05`):** Kedro + kedro-mlflow, MLflow, Great Expectations, Featuretools, Hopsworks, SHAP, FastAPI, Docker, Kubernetes (KIND/HPA), Evidently.
- **Key asset:** `week_05/bank-example` is a near-complete reference of this project; `week_05/spotify_recommender` covers container serving. **Risk:** copying without rebuilding hurts the creativity + learning marks → we rebuild node logic.
- **Tech debt to avoid:** hard Hopsworks (cloud) dependency breaks the reproducibility the brief demands → mitigation: configurable local feature store.
- **Quality target:** 20/20 — modular code, separately-runnable pipelines, full reproducibility, 6-page report with SHAP explainability.

**Approved Expert Squad (4):**

| Expert | Responsibility | Requirements covered |
|--------|----------------|----------------------|
| `@mlops-pipeline-architect` | Kedro structure, catalog, registry, orchestration | Kedro reqs + #6 |
| `@data-quality-expert` | Great Expectations, feature store, CI/CD gate, drift/Evidently | #1 + #5 |
| `@ml-experimentation-expert` | MLflow, SHAP, model selection, Optuna | #2 + #3 |
| `@serving-devops-expert` | FastAPI + Docker + K8s/HPA, tests, reproducibility | #4 |

---

## Context

This is a graded MSc MLOps capstone. The brief (`MLOps_project.pdf`) asks us to **simulate the real-world process of deploying an ML model** using the concepts from the 6 course weeks, delivered as:

- A **report (max 6 pages)**: why this data + success metrics; agile/sprint planning; EDA & modelling results (plots, feature importance, explainability); a production-readiness discussion (advantages/risks/mitigations); and a package+version list.
- A **Kedro-organized, modular, orchestrated pipeline** with separately runnable components, including: (1) unit data tests + feature store, (2) MLflow tracking + model versioning, (3) saved metrics + SHAP explainability, (4) model serving + containers, (5) data drift evaluation, (6) tests for relevant functions/pipelines.
- Delivered as report + zip (with runnable sample data) **or a Git link**.

Grading is on **quality of report, quality of code, and creativity** shown in using the technologies.

**Decisions locked with the user:**
- **Scope = Full stack (max marks):** Kedro + MLflow + SHAP + FastAPI-in-Docker + data drift (Evidently) + Kubernetes (optional stretch) + tests.

---

## Step 0 (DO FIRST): Pick the dataset — 5 recommendations

All five are tabular, have a clean target, tell a good "why this data / business metric" story, give a rich SHAP narrative, and let us *manufacture drift* for the drift component. Recommendation order = best fit for 20/20.

| # | Dataset | Task | Why it scores well | SHAP story | Drift story |
|---|---------|------|--------------------|-----------|-------------|
| **1 ⭐ Telco Customer Churn** (IBM/Kaggle, ~7k rows) | Binary classification (churn) | Clear business metric (retention $, recall on churners); small & fast; not the class bank dataset | Contract type, tenure, monthly charges dominate — intuitive | Shift tenure/charges in a batch to simulate a pricing change |
| **2 House Pricing** (Ames/Kaggle, ~1.5k rows) | Regression (sale price) | The PDF's *own example* references house pricing — signals you read the brief; strong regression EDA | Living area, quality, neighborhood | Inflate prices / shift a feature to mimic a market shift |
| **3 Credit Default** (UCI credit card clients, 30k rows) | Binary classification (default) | Finance realism; class imbalance → justifies F1/ROC-AUC over accuracy; bigger data → strong "Pandas won't scale, propose Spark" mitigation | Credit limit, payment history, bill amounts | Drift PAY_*/bill amounts to simulate downturn |
| **4 Online Shoppers Intention** (UCI, 12k rows) | Binary classification (purchase) | Web/behavioral angle = creativity; seasonality; clean | Page values, exit rates, visitor type | Shift traffic source / month (seasonal drift is natural) |
| **5 Heart Disease / Stroke** (UCI/Kaggle, ~5k rows) | Binary classification (disease) | Healthcare = high-stakes explainability narrative (why SHAP matters for trust) | Age, BP, cholesterol — clinically interpretable | Shift age/BP distribution to simulate a different cohort |

**DATASET LOCKED: #4 Online Shoppers Purchasing Intention** (UCI id=468) — chosen for a mix of creativity + reliability: novel e-commerce/behavioural angle, **natural seasonal drift** (req #5 is genuine, not simulated), single clean CSV (~12k rows) so the pipeline reaches green fast. Multi-table Featuretools/DFS kept as an optional stretch (derive session-aggregates per VisitorType/Month). See [data_download.md](data_download.md). Runner-ups: Telco Churn (max reliability) / Olist e-commerce (max creativity, multi-table).

---

## Target Project Structure (mirrors `week_05/bank-example`)

```
mlops_project/                         # Kedro project root (Git repo)
├── conf/
│   ├── base/
│   │   ├── catalog.yml                # datasets across data/01_raw → 08_reporting
│   │   ├── parameters.yml             # target col, split, model params
│   │   └── parameters_optuna.yml      # HPO search space (stretch)
│   └── local/
│       ├── credentials.yml            # Hopsworks keys (gitignored) — OR skip Hopsworks (see note)
│       └── mlflow.yml                 # kedro-mlflow config
├── data/                              # 01_raw … 08_reporting (Kedro convention)
├── docs/                              # report assets
├── notebooks/                         # EDA exploration (kept, per brief)
├── src/<package>/
│   ├── pipelines/
│   │   ├── data_quality/              # GX unit data tests  (Week 1)
│   │   ├── feature_store/             # Hopsworks ingest/retrieve OR local feature store (Week 1)
│   │   ├── data_cleaning/             # (Week 3/5)
│   │   ├── data_feat_engineering/     # featuretools/manual FE (Week 1/5)
│   │   ├── data_split/                # train/test split (Week 5)
│   │   ├── model_selection/           # compare candidates, pick champion (Week 5)
│   │   ├── model_train/               # train + MLflow autolog + SHAP (Week 2/3/5)
│   │   ├── model_predict/             # batch inference (Week 5)
│   │   └── data_drift/                # Evidently drift report (Week 6)
│   ├── pipeline_registry.py           # named + __default__ pipelines
│   └── settings.py
├── serving/                           # FastAPI + Docker (Week 4/5 spotify_recommender pattern)
│   ├── app/main.py                    # /predict, /health, /ready
│   ├── Dockerfile
│   └── (k8s: deployment.yml, service.yml, hpa.yml — stretch)
├── tests/                             # pytest for nodes + pipelines (Week 3/4)
├── pyproject.toml
└── README.md                          # how to run end-to-end
```

**Reference files to study/adapt (rebuild logic to learn, don't blind-copy):**
- Kedro registry: `Theory Classes/week_05/bank-example/src/bank_example/pipeline_registry.py`
- SHAP in training: `.../week_05/bank-example/src/bank_example/pipelines/model_train/nodes.py`
- MLflow config: `.../week_05/bank-example/conf/local/mlflow.yml`
- GX suites: `.../week_01/gx_project/gx/expectations/bank_quality_v1.json`
- Feature-store utils: `.../week_02/feature_utils.py` (`to_feature_store`, `validate_and_upload_features`, `build_expectation_suite`, `plot_correlation_with_target`)
- MLflow + Optuna: `.../week_02/01_MLflow_intro.ipynb`, `02_Optuna_MLFlow.ipynb`
- Serving: `.../week_05/spotify_recommender/app/main.py`, `Dockerfile`, `deployment.yml`, `hpa.yml`
- Drift output format: `.../week_05/bank-example/data/08_reporting/drift_result.csv`, `data_drift_report.html`

> **Note on the feature store:** the class uses Hopsworks (cloud, needs API key + the `C:/tmp` Windows fix). To keep the project *runnable by graders* (brief requires reproducibility), we make the feature store **configurable**: default to a local parquet "feature store" in `data/04_feature/`, with the Hopsworks path documented. This satisfies requirement #1 without a hard cloud dependency. Decide at Sprint 1.

---

## Verification (end-to-end — brief requires reproducible results)

1. Clean-env install from pinned versions (`pip install -e .` / `uv` / `requirements.txt`).
2. `kedro run` → full chain (`data_quality → … → model_train → data_drift`) completes; artifacts in `data/` + MLflow.
3. `kedro run --pipeline=data_quality` and `--pipeline=data_drift` run **independently** (brief requirement).
4. `mlflow ui` shows run with params, metrics, model, SHAP artifact.
5. `docker build` + `docker run` serving image; `curl /health` → 200; `POST /predict` → prediction.
6. `pytest` → all node/pipeline tests green.
7. Drift pipeline on manufactured drifted batch → flags drift.
8. README documents every command; sample data committed (or Git link) so a grader reproduces results.

---

## Collaboration model

Each sprint: I (1) explain the concept + point to the exact class file that teaches it, (2) we write the node/config together, (3) run + verify, (4) I flag production-grade vs. teaching shortcut. You drive implementation; I scaffold, explain, review. Commit at the end of each sprint.

**Immediate next action:** confirm your dataset pick (Step 0), then start Sprint 0 (`kedro new` + repo scaffold).

---

*See [tasks.md](tasks.md) for the sprint-by-sprint backlog as checkable tasks.*
