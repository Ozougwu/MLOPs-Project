# MSc MLOps Project — Merged Plan & Backlog (20/20 Delivery)

> Single merged document combining the **technical roadmap** ([implementation_plan.md](implementation_plan.md)) and the **sprint backlog** ([tasks.md](tasks.md)). Planned via the **Andru.ia Solutions Architect** skill. All content in **English** (user override; skill defaults to Spanish).

---

# PART 1 — ROADMAP

## 🤖 Andru.ia Diagnosis & Technical Prescription

**Scenario B — Evolution Project.** The workspace contains preexisting course material (6 weeks of theory, reference Kedro projects in `week_05`, practical notebooks). `Practical/` and `Main Project/` are empty: we build from scratch by **evolving** the patterns already taught.

**Technical Prescription:**
- **Available stack (all in `Theory Classes/week_05`):** Kedro + kedro-mlflow, MLflow, Great Expectations, Featuretools, Hopsworks, SHAP, FastAPI, Docker, Kubernetes (KIND/HPA), Evidently.
- **Key asset:** `week_05/bank-example` ≈ near-complete reference of this project; `week_05/spotify_recommender` covers container serving. **Risk:** copying without rebuilding hurts creativity + learning marks → we rebuild node logic.
- **Tech debt to avoid:** hard Hopsworks (cloud) dependency breaks reproducibility → mitigation: configurable local feature store.
- **Quality target:** 20/20 — modular code, separately-runnable pipelines, full reproducibility, 6-page report with SHAP explainability.

**Approved Expert Squad (4):**

| Expert | Responsibility | Reqs |![![![alt text](image-2.png)](image-1.png)](image.png)
|--------|----------------|------|
| `@mlops-pipeline-architect` | Kedro structure, catalog, registry, orchestration | Kedro + #6 |
| `@data-quality-expert` | Great Expectations, feature store, CI/CD gate, drift/Evidently | #1 + #5 |
| `@ml-experimentation-expert` | MLflow, SHAP, model selection, Optuna | #2 + #3 |
| `@serving-devops-expert` | FastAPI + Docker + K8s/HPA, tests, reproducibility | #4 |

## Context

Graded MSc MLOps capstone. The brief (`MLOps_project.pdf`) asks us to **simulate real-world ML deployment** using the 6 course weeks, delivered as:
- **Report (max 6 pages):** why this data + success metrics; agile planning; EDA & modelling results (plots, feature importance, explainability); production discussion (advantages/risks/mitigations); package+version list.
- **Kedro modular orchestrated pipeline**, separately runnable, including: (1) unit data tests + feature store, (2) MLflow tracking + versioning, (3) metrics + SHAP, (4) serving + containers, (5) data drift, (6) tests.
- Delivered as report + zip (with runnable sample data) **or a Git link**.

Grading = **report quality + code quality + creativity**. **Scope locked: Full stack (max marks).**

> **Required vs stretch policy (locked).** *Required* = anything the brief names in text or its pipeline diagram — the **9 pipelines** must each `kedro run --pipeline=<name>` **standalone** (brief: "choose just to run separated pipeline"). `model_predict` is **required** (in the diagram; feeds serving + drift), **not** a stretch. *Stretch* = raises the creativity ceiling only; **isolated + optional** so it can never break the "everyone produces the same results" clause. Add stretches **only after** the 9 pipelines + a solid report are done. Stretch priority: (1) natural seasonal drift story — nearly free, built into the required drift pipeline → (2) Optuna HPO (fixed candidates stay the default path) → (3) Kubernetes/KIND/HPA (Docker alone satisfies #4).

## Step 0 — Pick the dataset (5 recommendations)

| # | Dataset | Task | Why it scores well | SHAP story | Drift story |
|---|---------|------|--------------------|-----------|-------------|
| **1 ⭐ Telco Customer Churn** (~7k) | Binary (churn) | Clear business metric; small/fast; not the class bank dataset | Contract, tenure, monthly charges | Shift tenure/charges to mimic pricing change |
| **2 House Pricing** (Ames, ~1.5k) | Regression | The PDF's own example; strong regression EDA | Living area, quality, neighborhood | Inflate prices / shift feature → market shift |
| **3 Credit Default** (UCI, 30k) | Binary (default) | Imbalance → justifies F1/ROC-AUC; scale → "Pandas won't scale, propose Spark" | Credit limit, payment history | Drift PAY_*/bill amounts → downturn |
| **4 Online Shoppers Intention** (UCI, 12k) | Binary (purchase) | Behavioral angle = creativity; seasonality | Page values, exit rates | Shift traffic source / month |
| **5 Heart Disease / Stroke** (~5k) | Binary (disease) | High-stakes explainability narrative | Age, BP, cholesterol | Shift age/BP → different cohort |

**DATASET LOCKED: #4 Online Shoppers Purchasing Intention** (UCI id=468). Chosen for a mix of creativity + reliability: novel e-commerce angle with **natural seasonal drift** (req #5 isn't faked), single clean CSV (~12k rows). Multi-table Featuretools/DFS kept as an optional stretch. See [data_download.md](data_download.md). Runner-ups: Telco Churn (reliability) / Olist (max creativity).

## Target Project Structure (mirrors `week_05/bank-example`)

```
mlops_project/                         # Kedro project root (Git repo)
├── conf/base/{catalog.yml, parameters.yml, parameters_optuna.yml}
├── conf/local/{credentials.yml (gitignored), mlflow.yml}
├── data/                              # 01_raw … 08_reporting
├── docs/                              # report assets
├── notebooks/                         # EDA exploration (kept, per brief)
├── src/<package>/pipelines/
│   ├── data_quality/                  # GX unit data tests        (Wk1, #1)
│   ├── feature_store/                 # local store / Hopsworks    (Wk1, #1)
│   ├── data_cleaning/                 # (Wk3/5)
│   ├── data_feat_engineering/         # featuretools/manual FE     (Wk1/5)
│   ├── data_split/                    # train/test split           (Wk5)
│   ├── model_selection/               # champion selection         (Wk5)
│   ├── model_train/                   # train + MLflow + SHAP       (Wk2/3/5, #2/#3)
│   ├── model_predict/                 # batch inference (REQUIRED) (Wk5)
│   └── data_drift/                    # Evidently drift report     (Wk6, #5)
├── src/<package>/{pipeline_registry.py, settings.py}
├── serving/{app/main.py, Dockerfile, k8s manifests (stretch)}     # (Wk4/5, #4)
├── tests/                             # pytest nodes + pipelines    (Wk3/4, #6)
├── pyproject.toml
└── README.md
```

**Reference files to study/adapt (rebuild logic, don't blind-copy):**
- Kedro registry: `week_05/bank-example/src/bank_example/pipeline_registry.py`
- SHAP training: `week_05/bank-example/.../model_train/nodes.py`
- MLflow config: `week_05/bank-example/conf/local/mlflow.yml`
- GX suites: `week_01/gx_project/gx/expectations/bank_quality_v1.json`
- Feature-store utils: `week_02/feature_utils.py` (`to_feature_store`, `validate_and_upload_features`, `build_expectation_suite`, `plot_correlation_with_target`)
- MLflow + Optuna: `week_02/01_MLflow_intro.ipynb`, `02_Optuna_MLFlow.ipynb`
- Serving: `week_05/spotify_recommender/{app/main.py, Dockerfile, deployment.yml, hpa.yml}`
- Drift output format: `week_05/bank-example/data/08_reporting/{drift_result.csv, data_drift_report.html}`

> **Feature store note:** class uses Hopsworks (cloud + `C:/tmp` Windows fix). For grader reproducibility we make it **configurable** — default to local parquet in `data/04_feature/`, document the Hopsworks path. Decide at Sprint 1.

---

# PART 2 — SPRINT BACKLOG

*Requirement legend: #1 data tests + feature store · #2 MLflow · #3 metrics + SHAP · #4 serving + containers · #5 drift · #6 tests.*

### Sprint 0 — Setup, Artifacts & Data ✅ DONE
- [x] Create Andru.ia artifacts `tasks.md` + `implementation_plan.md` (English)
- [x] Confirm dataset — **Online Shoppers Purchasing Intention** (locked; see [data_download.md](data_download.md))
- [x] Download `online_shoppers_intention.csv` → `data/01_raw/online_shoppers.csv`
- [x] `kedro new` + init Git
- [x] Set up `pyproject.toml`, `conf/`, `data/` (01_raw…08_reporting)
- [x] Raw data → `data/01_raw/`; rebuilt EDA notebook (`notebooks/01_eda.ipynb`, executes clean)

### Sprint 1 — Data Quality + Feature Store *(#1, Wk1)* ✅ DONE
- [x] `data_quality`: GX suite (**8 asserts**) + validation gate node (raises `ValueError`)
- [x] `data_feat_engineering`: 7 per-row engineered features
- [x] `feature_store`: persist to local parquet feature groups (Hopsworks documented, not required)
- [ ] Verify `kedro run --pipeline=data_quality` independently *(deferred to Sprint 5 separated-run proof)*

### Sprint 2 — Clean, Split, Train + MLflow + SHAP *(#2 & #3, Wk2/3/5)* ✅ DONE
- [x] `data_cleaning`, `data_split`, `model_selection`, `model_train` (full chain runs, exit 0, 9/9 tasks)
- [x] `model_predict` **(required pipeline)**: loads registered champion → scores 2,466 rows → `07_model_output`
- [x] `kedro-mlflow`: autolog + manual metrics; champion **registered** `online_shoppers_champion` (v2 on re-run → #2 versioning proven)
- [x] SHAP summary plot (TreeExplainer, ~1 min) logged as MLflow artifact + `08_reporting`; metrics JSON → `08_reporting`
- [x] **Result:** champion = RandomForest, test ROC-AUC **0.927**, recall(buyer) **0.73**; SHAP confirms PageValues #1 (matches EDA) + `has_page_value` #2 (validates FE)
- [ ] *(Stretch — isolated/optional)* Optuna HPO; fixed candidates stay the default path
- [ ] *(Report-in-parallel)* metric/SHAP artifacts now in `08_reporting` — pull into report at Sprint 6

### Sprint 3 — Serving + Containers *(#4, Wk4/5)* ✅ DONE
- [x] FastAPI `serving/app/main.py`: `/predict`, `/health`, `/ready` (loads portable champion + feature schema)
- [x] `Dockerfile` (python:3.12-slim, uvicorn, **non-root**, uv, HEALTHCHECK); image built (1.18 GB)
- [x] **Verified in-container:** `/health` 200, `/ready` 200, `/predict` outputs **match the pipeline exactly** (0.00079 / 0.14437 / 0.79601)
- [x] **Reproducibility fix:** registry stored an absolute Windows path → model failed to load in the Linux container. Added a portable local MLflow model dir (`data/06_models/champion_model`, no abs paths) that serving loads by default; registry alias kept as opt-in. Now runs on any machine.
- [ ] *(Stretch — isolated/optional)* k8s + KIND + HPA; clearly labeled "not required to run"

### Sprint 4 — Data Drift *(#5, Wk6)*
- [ ] `data_drift` with **Evidently**: reference vs. current → HTML + CSV in `08_reporting`
- [ ] **Natural drift story (creativity, ≈free):** reference = low-season months, current = Nov holiday batch (Feb ~1.6% → Nov ~25% buyers)
- [ ] Manufacture drifted sample; verify `--pipeline=data_drift` independently

### Sprint 5 — Tests + Orchestration + Reproducibility *(#6, Wk3/4)*
- [ ] `pytest` for key nodes + ≥1 pipeline test
- [ ] `pipeline_registry.py`: named pipelines + `__default__`; each runs separately AND full chain
- [ ] Pin versions; fresh-clone reproducibility

### Sprint 6 — Report (max 6 pages)
- [ ] Data choice + success metrics · project planning · EDA/modelling (plots, importance, SHAP)
- [ ] Production discussion (advantages/risks/mitigations incl. Pandas→Spark) · package+version list
- [ ] Final README; sample data or Git link

---

# PART 3 — VERIFICATION (Definition of Done)

1. Clean-env install from pinned versions.
2. `kedro run` → full chain completes; artifacts in `data/` + MLflow.
3. **Separated-run proof:** every one of the 9 pipelines runs **independently** via `kedro run --pipeline=<name>` — demonstrated and documented in the README command table (brief: "choose just to run separated pipeline").
4. `mlflow ui` shows run with params, metrics, model, SHAP artifact.
5. `docker build` + `docker run`; `curl /health` → 200; `POST /predict` → prediction.
6. `pytest` green.
7. Drift pipeline flags a manufactured drifted batch.
8. README documents every command; sample data committed (or Git link) → grader reproduces results.

---

## Collaboration Model

Each sprint: I (1) explain the concept + point to the exact class file that teaches it, (2) we write the node/config together, (3) run + verify, (4) I flag production-grade vs. teaching shortcut. You drive implementation; I scaffold, explain, review. Commit at the end of each sprint.

**Immediate next action:** confirm your dataset pick (Step 0), then start Sprint 0 (`kedro new` + repo scaffold).
