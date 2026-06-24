# Tasks Backlog — MSc MLOps Project (Agile Sprints)

> Detailed backlog for the 20/20 delivery. Each sprint is a **collaborative learning session**. Tick tasks as we complete them. See [implementation_plan.md](implementation_plan.md) for the full technical roadmap.

**Requirement legend (from `MLOps_project.pdf`):** #1 data tests + feature store · #2 MLflow tracking/versioning · #3 metrics + SHAP · #4 serving + containers · #5 data drift · #6 tests.

---

## Sprint 0 — Setup, Artifacts & Data
*Goal: working Kedro skeleton + project artifacts. Learn: Kedro anatomy, catalog, parameters.*

- [x] Create Andru.ia artifacts `Main Project/tasks.md` + `Main Project/implementation_plan.md` (English)
- [x] Confirm dataset — **Online Shoppers Purchasing Intention** (locked; see [data_download.md](data_download.md))
- [ ] Download `online_shoppers_intention.csv` → `data/01_raw/online_shoppers.csv`
- [ ] `kedro new` and initialise Git repo
- [ ] Set up `pyproject.toml`, `conf/base/`, `conf/local/`, `data/` (01_raw…08_reporting)
- [ ] Drop raw data into `data/01_raw/`
- [ ] Quick EDA notebook in `notebooks/`

## Sprint 1 — Data Quality + Feature Store *(Req #1, Week 1)*
*Goal: validated, engineered, stored features. Learn: data contracts, CI/CD gate, feature versioning.*

- [ ] `data_quality` pipeline: Great Expectations suite with **≥6 asserts**
- [ ] Validation gate node (pass → continue / fail → halt)
- [ ] `data_feat_engineering` pipeline: engineer features
- [ ] `feature_store` pipeline: persist features to local parquet store (Hopsworks optional/configurable)
- [ ] Verify `kedro run --pipeline=data_quality` runs independently

## Sprint 2 — Clean, Split, Train + MLflow + SHAP *(Req #2 & #3, Weeks 2/3/5)*
*Goal: tracked, explainable champion model. Learn: tracking, model versioning, explainability.*

- [ ] `data_cleaning` pipeline
- [ ] `data_split` pipeline (train/test)
- [ ] `model_selection` pipeline (compare candidates → champion)
- [ ] `model_train` pipeline with `kedro-mlflow` (autolog, params, metrics, model artifact, signature)
- [ ] SHAP summary plot logged as MLflow artifact
- [ ] Save main metrics to `data/08_reporting/`
- [ ] *(Stretch)* Optuna HPO via `parameters_optuna.yml`

## Sprint 3 — Serving + Containers *(Req #4, Weeks 4/5)*
*Goal: containerised prediction service. Learn: model packaging, container serving, probes.*

- [ ] FastAPI `serving/app/main.py` loading champion model
- [ ] Endpoints: `/predict`, `/health`, `/ready`
- [ ] `Dockerfile` (python-slim, uvicorn, non-root user)
- [ ] Build + run container; smoke-test `/health` and `/predict`
- [ ] *(Stretch)* k8s manifests + KIND cluster + HPA

## Sprint 4 — Data Drift *(Req #5, Week 6)*
*Goal: drift monitoring component. Learn: drift detection, silent degradation.*

- [ ] `data_drift` pipeline with **Evidently**: reference vs. current batch
- [ ] Output drift report HTML + result CSV to `data/08_reporting/`
- [ ] Manufacture a drifted sample to demonstrate detection
- [ ] Verify `kedro run --pipeline=data_drift` runs independently

## Sprint 5 — Tests + Orchestration + Reproducibility *(Req #6, Weeks 3/4)*
*Goal: tested, reproducible, modular pipeline. Learn: pipeline testing, modular orchestration.*

- [ ] `pytest` tests for key nodes (cleaning, FE, drift logic)
- [ ] At least one full-pipeline test
- [ ] `pipeline_registry.py`: named pipelines + `__default__` full chain
- [ ] Verify each pipeline runs separately AND the full sequence runs
- [ ] Pin package versions; confirm fresh-clone reproducibility

## Sprint 6 — Report (max 6 pages)
*Goal: the graded report. Learn: communicating an MLOps system.*

- [ ] Data choice + success metrics
- [ ] Project planning (this sprint breakdown)
- [ ] EDA & modelling results (plots, feature importance, SHAP)
- [ ] Production discussion (advantages / risks / mitigations — incl. "Pandas won't scale → propose Spark")
- [ ] Package + version list
- [ ] Final README with all run commands; sample data or Git link

---

## Definition of Done (whole project)

- [ ] `kedro run` reproduces all results end-to-end on a clean env
- [ ] Each pipeline runs independently (`--pipeline=<name>`)
- [ ] MLflow UI shows tracked run with metrics, model, SHAP
- [ ] Serving container responds to `/predict`
- [ ] Drift pipeline flags a manufactured drifted batch
- [ ] `pytest` green
- [ ] Report ≤ 6 pages, all required sections present
