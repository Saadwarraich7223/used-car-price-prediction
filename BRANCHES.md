# Branch Map — Used Car Price Prediction

All project work lives on topic branches, one logical unit per branch, pushed to
`origin`. Merge them into `main` **in the order below** (each branch depends on
the ones before it). Use the suggested PR title and merge description for each.

**Base:** `main` at `42778b0` (Merge PR #9 — feature/model-tuning)

---

## 1. `fix/test-create-features`

- **Head commit:** `1247d38`
- **PR title:** `Fix test_create_features to match feature engineering implementation`
- **Contains:** `tests/test_features.py` (rewritten fixture + assertions)

The existing `test_create_features` fixture lacked the `name` column and expected
values that did not match `create_features` (it used 4 arbitrary rows; the real
function computes `vehicle_age`, `km_per_year`, `power_per_cc`, `brand`).

**Merge description:**
> Fixes the broken `test_create_features` test which failed with `KeyError:
> 'km_driven'` because its fixture did not include the `name` column and asserted
> values inconsistent with `src/features/engineering.py`.
> The fixture is now a 2-row DataFrame with all required columns, and the
> assertions match the real implementation (verified: 4/4 tests pass, including
> `tests/test_cleaning.py`).

---

## 2. `chore/project-config`

- **Head commit:** `3efc516`
- **PR title:** `Configure project metadata and curate dependency files`
- **Contains:** `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `.gitignore`

Adds PEP 621 project metadata + `[tool.ruff]` config, a curated runtime
`requirements.txt`, a `requirements-dev.txt` (runtime + pytest/ruff/notebook
tooling), and a `.gitignore` covering `.venv`, notebooks checkpoints, and
generated artifacts.

**Merge description:**
> Makes the project installable and documented at the tooling level: full
> project metadata and ruff settings in `pyproject.toml`, split runtime vs dev
> dependencies, and a `.gitignore` that keeps `data/processed/` and `models/`
> out of version control.

---

## 3. `ci/github-actions`

- **Head commit:** `8b5f188`
- **PR title:** `Add CI workflow with linting and tests`
- **Contains:** `.github/workflows/ci.yml`

GitHub Actions workflow: checkout, Python 3.13, pip cache, install
`requirements-dev.txt`, run `ruff check src tests` and `pytest -v`.

**Merge description:**
> Adds CI that runs on push/PR: ruff linting of `src/` and `tests/` plus the
> pytest suite on Python 3.13, so every merge is gated on green checks.

---

## 4. `feature/model-validation`

- **Head commit:** `488942a`
- **PR title:** `Finalize best model, persist pipeline artifact, and document validation results`
- **Contains:** `notebooks/04_model_experiments.ipynb` (3 new cells appended)

Adds the final model-selection step: trains the tuned Random Forest
(`n_estimators=500, min_samples_split=5, min_samples_leaf=1, max_features=1.0,
random_state=42`) on the engineered features, reports test metrics, and persists
the full pipeline to `models/car_price_pipeline.joblib` (gitignored).

**Merge description:**
> Locks in the production model. Random Forest achieves **R² = 0.9321,
> RMSE = Rs 122,077, MAE = Rs 70,738** on the held-out split (1,386 cars),
> beating the linear baselines (R² ≈ 0.87-0.89) and Gradient Boosting
> (R² = 0.9254). The trained pipeline is saved as
> `models/car_price_pipeline.joblib`. Notebook re-executed end-to-end with zero
> errors (nbconvert + verified no tracebacks).

---

## 5. `feature/inference-pipeline`

- **Head commit:** `9ceabb8` (also `86956d3`)
- **PR title:** `Add production inference pipeline modules`
- **Contains:** `configs/model_config.yaml`, `src/models/{train,predict,evaluate}.py`,
  `src/models/__init__.py`, `src/utils/{__init__,config}.py`,
  `src/visualization/{__init__,plots}.py`, `tests/test_predict.py`, `.gitignore`

Productionizes the notebook pipeline as importable, CLI-runnable modules
(architecture is copied verbatim from the validated notebook):
- `python -m src.models.train` — retrain + persist artifact (env override
  `MODEL_N_ESTIMATORS` supported)
- `python -m src.models.predict --input '<json>'` — one-off prediction
- `python -m src.models.evaluate` — re-score the persisted artifact
- `configs/model_config.yaml` — single source of truth for features/params/paths
- `.gitignore` change: `models/` anchored to `/models/` so `src/models/` is tracked

**Merge description:**
> Turns the validated notebook model into a reusable package. The canonical
> preprocessing (median/most-frequent imputation, StandardScaler,
> OneHotEncoder) and the tuned Random Forest are wrapped in `src/models/` with
> train/predict/evaluate entry points driven by `configs/model_config.yaml`.
> Verified: 4/4 new unit tests pass; smoke train reproduces notebook metrics
> (R² = 0.9331 at 50 trees); smoke predict returns Rs 532,008 for a real
> Maruti Swift Dzire. Also anchors the `models/` gitignore pattern to the repo
> root so `src/models/` source code is tracked.

> **Merge note (conflict expected):** this branch and `chore/project-config`
> both touch `.gitignore` — a conflict may occur. Resolve by keeping BOTH the
> `.omo/` entry (from chore) and the `/models/` root-anchored entry (from this
> branch).

---

## 6. `feature/prediction-app`

- **Head commit:** `d2b69dd`
- **PR title:** `Add Streamlit demo app for price prediction`
- **Contains:** `app.py`

Interactive Streamlit app: form for all 13 raw attributes → `predict_price()` →
price displayed in Indian Lakh/Crore notation. Handles a missing artifact
gracefully with a retrain hint.

**Merge description:**
> Adds a `streamlit run app.py` demo UI. Users fill in car specs and get a live
> price estimate from the persisted artifact (imports `src.models.predict` from
> the inference-pipeline branch, which must be merged first). Verified end-to-end
> against the real 500-tree artifact: Swift Dzire 2014 → Rs 5.21 Lakh.

---

## 7. `docs/readme`

- **Head commit:** `77b1071`
- **PR title:** `Add project README with setup, usage, and results`
- **Contains:** `README.md`

Project README: dataset overview, project structure, setup, CLI + app usage,
notebook index, verified results table, and branch-layout pointer.

**Merge description:**
> Documents the whole project: how to install, retrain, predict, evaluate, run
> the app, and read the notebooks, with the verified model comparison table
> (Random Forest R² 0.9321 vs linear baselines) and top feature importances.

---

## 8. `docs/final-report`

- **Head commit:** `60306f2`
- **PR title:** `Add final report with model results and figures`
- **Contains:** `reports/final_report.md`, `reports/figures/{residuals,price_distribution}.png`

End-to-end report: objective, dataset, cleaning/feature engineering methodology,
model comparison, results with figures generated from the real artifact,
productionization notes, conclusions and future work.

**Merge description:**
> Comprehensive final report covering the full pipeline from raw Cardekho data
> (8,128 listings, 6,926 usable) to the production Random Forest. Figures are
> generated from the actual 500-tree artifact (residuals vs predicted on the
> held-out split; price distribution). Documents strengths (R² 0.9321,
> MAE Rs 70,738), the premium-car limitation, and future work (SHAP, FastAPI).

---

## 9. `docs/branch-context` (this branch)

- **Head commit:** (this commit)
- **PR title:** `Add branch map with PR titles and merge descriptions`
- **Contains:** `BRANCHES.md`

This file. A single map of every branch above with its PR title, merge
description, and the recommended merge order.

**Merge description:**
> Adds BRANCHES.md — the authoritative map of all project branches with PR
> titles, merge descriptions, dependency order, and known merge-conflict notes
> (`.gitignore` between chore/project-config and feature/inference-pipeline).
> Merge last so the map reflects the final state.

---

## Merge order summary

| Order | Branch | PR title |
|---|---|---|
| 1 | `fix/test-create-features` | Fix test_create_features to match feature engineering implementation |
| 2 | `chore/project-config` | Configure project metadata and curate dependency files |
| 3 | `ci/github-actions` | Add CI workflow with linting and tests |
| 4 | `feature/model-validation` | Finalize best model, persist pipeline artifact, and document validation results |
| 5 | `feature/inference-pipeline` | Add production inference pipeline modules |
| 6 | `feature/prediction-app` | Add Streamlit demo app for price prediction |
| 7 | `docs/readme` | Add project README with setup, usage, and results |
| 8 | `docs/final-report` | Add final report with model results and figures |
| 9 | `docs/branch-context` | Add branch map with PR titles and merge descriptions |
