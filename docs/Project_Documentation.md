## BCA EU Case Study – End-to-End ML System Documentation

Author: Jonathan Lampkin
Author email: jmlampkin@gmail.com

Repository: [GitHub – BCA_EU_Case_Study](https://github.com/jonathanlampkin/BCA_EU_Case_Study)


### 1) Purpose and Scope
This repository implements a robust, time-aware, end-to-end machine learning system for predicting used car prices. It includes data preprocessing, feature engineering, multicollinearity pruning, model comparison with time series cross-validation, hyperparameter optimization, SHAP-based interpretability, experiment tracking, and a FastAPI serving layer. The project was built quickly (within two days) with AI assistance due to the short time window; the system design, direction, and all decisions were led by me (Jonathan Lampkin). I am familiar with and accountable for every component. This project was very rushed and not everything is complete and the code is not optimal and at times over-engineered. I would love to spend more time revising this codebase to finish implementing testing, monitoring, serving, and refactoring the code for efficiency and interpretability. Despite the rush, I hope you like the project and have an understanding that it was rushed and is not complete.


### 2) Quickstart – Getting Up and Running
Follow these steps to run locally.

- Clone the repo:
  - git clone https://github.com/jonathanlampkin/BCA_EU_Case_Study
  - cd BCA_EU_Case_Study

- Explore available commands:
  - make help

- Create and activate a Python environment (Python 3.12 recommended):
  - make setup
  - Or manually: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

- Recommended exploration flow:
  - Review the notebook: notebooks/eda.ipynb
  - Run end-to-end: make full (validate → preprocess → model → artifacts)
  - Or step-by-step:
    - make preprocess
    - make model
    - make serve (starts the FastAPI service)

- Optional services (monitoring/MLflow):
  - MLflow UI: launched by pipeline and logs saved under mlruns/
  - Prometheus/Grafana: docker-compose.monitoring.yml (see monitoring/)


### 3) High-level System Overview and Execution Order
The system follows this sequence:

1. Data Validation (optional)
2. Preprocessing and Feature Engineering (src/price_model/preprocessor.py)
3. Feature Selection / Multicollinearity Pruning (src/price_model/advanced_feature_selector.py)
4. Model Comparison via TimeSeriesSplit CV (src/price_model/modeling_pipeline.py)
5. Hyperparameter Optimization for best model (Optuna Bayesian Optimization)
6. Final Model Training + Explainability (SHAP) (src/price_model/feature_analyzer.py)
7. Artifact saving (model, preprocessor, reports, plots)
8. Serving API (src/price_model/model_service.py)

Where to look first:
- src/price_model/modeling_pipeline.py – orchestrates the pipeline
- src/price_model/preprocessor.py – all data preparation logic
- src/price_model/feature_analyzer.py – interpretability outputs
- src/price_model/model_service.py – runtime inference API


### 4) Exploratory Data Analysis (EDA) – Summary
Data: data/EUDS_CaseStudy_Pricing.csv, Shape: 18,575 rows × 19 columns

Key notes (artifacts/eda_summary.txt):
- Low missingness; notable: colour has ~0.10% missing.
- Numeric ranges are sensible but include outliers, e.g. cubeCapacity can be 0, kilometers, invalid chronological order between saledate and other date columns
- Target (targetPrice) is right-skewed; wide range.

Artifacts:
- artifacts/clean_eda/: distributions, correlation heatmap, text summary


### 5) Preprocessing and Feature Engineering
Primary module: src/price_model/preprocessor.py

Core components:
- DomainConfig: declarative configuration for numeric and categorical handling. Data Scientists can use this to pass in domain knowledge to be used in preprocessing.
  - numeric_cfg (Dict[str, ColCfg]) from src/price_model/num_sanitize.py controls min/max bounds declared from domain knowledge and imputation (median by default and does not leak data by only computing median using previous days) for numeric columns such as kilometers, cubeCapacity, powerKW, cylinder, vehicle_age_years, km_per_year, years_since_intro_at_sale.
  - categorical_cfg defines ordinal mappings, top-cumulative quantile OHE, and a high-cardinality threshold beyond which time-series target encoding is used.

- Preprocessor:
  - fit(X):
    - Applies stateless cleaning: date parsing, deduplication, doorNumber cleanup, chronology guard, cubeCapacity invalid→NaN, derived features (dates/mileage), composite make_model, vehicleID drop, text normalization.
    - Fits NumericSanitizer on present numeric columns to learn medians for imputation; collects z-score stats if enabled.
  - transform(X):
    - Re-applies stateless steps; imputes with learned medians; recomputes km_per_year; encodes categoricals per categorical_cfg; finalizes by dropping all-NA/zero-variance, ensuring numeric stability; optional z-score normalization.

Key transformers (src/price_model/transformers.py):
- CleanSaleDate: coerce saleDate to datetime; add sale_year.
- CleanDoorNumber: numeric coercion.
- YearChronologyGuard: centralizes chronology logic – drops rows with yearIntroduced > sale_year; masks implausible years (<1900); derives years_since_intro_at_sale.
- DateAndMileage: sale_year/month, vehicle_age_years, km_per_year.
- MakeModelComposite: combines make and model into make_model.
- DropVehicleId: drops vehicleID (kept simple and explicit).

Numeric sanitization (src/price_model/num_sanitize.py):
- ColCfg(min_val, max_val, replace_with) – per-column rules.
- NumericSanitizer.fit learns medians/means for configured numeric columns; transform applies bounds and imputation (leakage-safe: fit on train, apply on test).

Categorical handling and leakage-safe target encoding:
- Ordinal encoding via explicit mappings when appropriate (e.g., aesthetic/mechanical grades).
- One-hot encoding of most frequent categories up to a top cumulative quantile (k-1 strategy to avoid dummy trap).
- If cardinality>threshold and both targetPrice and saleDate exist (i.e., training context), compute a time-series target encoding using only past information: cumulative sums/counts per category minus the current row, smoothed toward overall prior with alpha=10.0. At inference, targetPrice is absent, so this path is skipped.


### 6) Feature Selection and Multicollinearity Pruning
Module: src/price_model/advanced_feature_selector.py
- Conservative pruning with correlation threshold and VIF threshold to reduce redundancy and instability.
- Produces a curated set of features for modeling and logs summary.


### 7) Modeling and Evaluation
Module: src/price_model/modeling_pipeline.py

Data preparation:
- load_and_prepare_data: reads artifacts/clean_data.csv (or configured path), converts saleDate to datetime, sorts by saleDate, separates X and y.

Model set:
- Tree-based: LightGBM, XGBoost, RandomForest; plus ensemble models (Voting, Stacking with Ridge).

Cross-validation:
- TimeSeriesSplit (cv_folds configurable) to respect temporal order and prevent leakage.

Metrics:
- RMSE, MAE, R², plus % metrics (MAPE/SMAPE/WMAPE) and quantiles of error distribution.

Model selection:
- Evaluate all models under TSCV; choose best by RMSE. Log to MLflow via ExperimentTracker.

Hyperparameter optimization (post-comparison):
- Optuna optimizes only the best-performing base model to save time and avoid overfitting selection bias.
- Re-evaluated with TimeSeriesSplit during optimization for consistent criteria.

Final training and artifacting:
- Retrain best model on full selected feature set; compute and save:
  - Model: artifacts/final_model.joblib (or improved_final_model.joblib)
  - Preprocessor/selector: artifacts/preprocessor.joblib (or improved_preprocessor.joblib)
  - Transformers snapshot (e.g., target transformer/scaler): artifacts/transformers.joblib
  - Results JSONs and plots under artifacts/


### 8) Explainability (SHAP) and Feature Insights
Module: src/price_model/feature_analyzer.py

Approach:
- For tree models, use shap.TreeExplainer with an independent masker on a small background sample (≤500) and interventional perturbation. Explain a sample of rows (default 1000) for global plots. Use a non-interactive plotting backend to avoid GUI/threading issues.

Artifacts:
- artifacts/feature_importance.png – model-based importance ranking
- artifacts/shap_summary.png – global SHAP beeswarm
- artifacts/shap_waterfall.png – local explanation of an instance
- artifacts/feature_interactions.png – SHAP-based interaction heatmap (correlation proxy)
- artifacts/feature_importance_ranking.csv – ranked feature list

Interpretation guide:
- Global: mean absolute SHAP shows overall drivers; beeswarm shows directionality (color = feature value; position = contribution sign/magnitude).
- Local: waterfall plot decomposes a single prediction into additive feature contributions around the baseline.


### 9) Serving – FastAPI Inference Service
Module: src/price_model/model_service.py

Key points:
- Loads trained model and selected-feature list; aligns incoming data columns (adds missing features as zeros, orders columns).
- Uses the same preprocessing pipeline principles (date fields, derived features). For target-encoding, the training-only path is skipped at inference because targetPrice is not present.
- Endpoints:
  - GET / – HTML landing page with examples
  - GET /health – health check
  - POST /predict – single prediction (Pydantic schema)
  - POST /predict/batch – CSV upload for batch predictions

Run locally:
- make serve
- Browser: http://localhost:8000/docs


### 10) Experiment Tracking and Artifacts
Module: src/price_model/experiment_tracker.py; outputs under mlruns/
- Logs parameters, metrics, and artifacts for reproducibility.
- Store plots and JSON summaries under artifacts/.


### 11) Data Leakage Controls
- Chronology guard: drops/repairs impossible temporal rows; derives years_since_intro_at_sale.
- Time-series CV: TimeSeriesSplit for evaluation.
- Target encoding: past-only cumulative statistics and overall prior smoothing; branch disabled at inference.
- Train/test separation: imputation stats learned on train only via NumericSanitizer.


### 12) Development, Testing, and Quality
Commands:
- make test – runs unit/integration/health checks in tests/*
- make lint / make format – code quality

Future enhancements:
- More granular unit tests per transformer and sanitizer
- Test fixtures for date parsing and chronology edge cases
- Smoke tests for API payload compatibility


### 13) Monitoring and Observability
Monitoring stack:
- Prometheus config in monitoring/prometheus.yml
- Grafana datasource in monitoring/grafana/
- docker-compose.monitoring.yml to bring up Prometheus/Grafana

Metrics:
- FastAPI: request count/duration, prediction metrics (count, duration, values)
- Extend with model drift and data quality monitors in future iterations


### 14) CI/CD and Deployment (Roadmap)
- CI: add GitHub Actions workflow for linting, tests, build (see .github/workflows/)
- CD: containerize and deploy via Docker/K8s; Azure YAML provided for guidance
- Weekly retraining: scheduled job to rerun make full on fresh data, compare metrics, and promote if improved


### 15) External Data Ideas
- Macro: fuel price indices, inflation, interest rates
- OEM/model release cadence; recall/maintenance indices
- Market signals: listing volume, regional demand indices
- Weather/seasonality proxies


### 16) Design Notes and Philosophy
- Separation of concerns: preprocessing vs. selection vs. modeling vs. serving
- Leakage avoidance across all stages
- Conservative feature selection and transparent explainability
- Simplicity where possible; explicit transformers to ease testing and iteration


### 17) Reproducible End-to-End Run
1. make setup
2. make full (or: make preprocess → make model)
3. Inspect artifacts/* and mlruns/*
4. make serve and query http://localhost:8000/docs


### 18) Credits
This system was designed end-to-end by Jonathan Lampkin. AI tools were used to accelerate implementation under a tight timeline; all critical design, domain assumptions, and final decisions were directed and reviewed by me.


