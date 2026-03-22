# big_data_project_2026
IMDB Big Data Project: Group 5
## Overview
This repository contains a production-grade machine learning pipeline designed to predict movie ratings using the IMDb dataset. The project integrates distributed data processing (PySpark), high-performance analytical querying, and advanced gradient boosting (XGBoost) within a modular, Object-Oriented architecture.

## Architecture & Key Features
The pipeline is divided into cleanly separated phases with explicit Parquet checkpoints. A **single SparkSession** with legacy Parquet format enabled is shared across all phases for efficiency and consistency.

1.  **Phase 1 — Ingestion + Cleaning:** Schema-on-read CSV/JSON ingestion via PySpark, null standardisation, text normalisation, numeric casting, and MAD (Hampel X84) outlier **clamping** (Winsorization) on all integer columns — extreme values are capped to the ±4× scaled-MAD boundary rather than dropped. (Script: `src/ingestion.py`)
2.  **Phase 2 — TMDB Enrichment + Feature Engineering:** Concurrent TMDB API hydration (10-worker ThreadPoolExecutor with semaphore rate limiter), runtime coalescing, and fingerprint-keyed normalization of `tmdb_primary_genre`, `tmdb_original_language`, `tmdb_origin_country`, and `tmdb_production_company` (with frequency threshold — rare companies collapsed to "unknown"). Interaction features (`budget_revenue_ratio`, `votes_x_popularity`) are engineered. XGBoost's native categorical support handles categoricals directly — no ordinal encoding. (Script: `src/tmdb_enrichment.py`)
3.  **Phase 3 — Graph Feature Computation:** Bipartite degree centralities, writer-writer synergy weights, and writer-director cross-role collaboration weights. DataFrames are repartitioned by `tconst` before self-joins to reduce shuffle volume. (Script: `src/graph_features.py`)
4.  **Deep Imputation (Optional):** MLP-based multivariate imputation of missing numeric values. (Script: `src/imputation.py`)
5.  **Bayesian Optimization:** Optuna-tuned XGBoost hyperparameters (`n_estimators`, `learning_rate`, `max_depth`, `reg_alpha`, `reg_lambda`), optimised for ROC-AUC via 3-fold Stratified CV with early stopping over 25 trials. (Script: `src/modeling.py`)
6.  **MLOps & Drift Governance:** K-S Test, PSI, and KL Divergence for covariate drift detection. (Script: `src/mlops.py`)
7.  **Global Experimentation Engine:** Grid search evaluator across imputer hyperparameters. Tags the winning model as `SUPREME_WINNER_MODEL.joblib` with full Optuna-optimized params in `SUPREME_WINNER_CONFIG.json`. Generates an **out-of-fold** misclassification report (`misclassified_examples.csv`) using honest CV predictions. (Script: `run_experiments.py`)
8.  **Unseen Inference Engine:** End-to-end prediction pipeline with isolated Parquet outputs, per-target TMDB caching, and shared SparkSession. Timestamped submissions prevent overwrites. (Script: `inference.py`)

## Parquet Checkpoint Flow
```
Phase 1 (Ingestion + Cleaning)
  → output/parquet/directing.parquet            (JSON crew — Spark native)
  → output/parquet/writing.parquet              (JSON crew — Spark native)
  → output/parquet/cleaned_data.parquet         (MAD-clamped — Spark native)

Phase 2 (TMDB Enrichment)
  → output/parquet/tmdb.parquet                 (TMDB API results — Spark native)
  → output/parquet/tmdb_enriched.parquet        (movies + TMDB + normalized categoricals)

Phase 3 (Graph Features)
  → output/parquet/enriched_features.parquet    (final feature matrix — Pandas-written)

Experiments
  → output/parquet/imputed_features.parquet     (if imputation enabled)
  → output/models/SUPREME_WINNER_MODEL.joblib   (winning model)
  → output/models/SUPREME_WINNER_CONFIG.json    (macro config + Optuna-optimized XGBoost params)

Inference
  → output/inference_parquet/tmdb_{target}.parquet  (per-target TMDB cache)
  → output/submissions/*_predictions_YYYYMMDD_HHMMSS.txt
```

## XGBoost Feature Schema
The pipeline maps the following 21 features into the `XGBClassifier`:

**Base IMDb Features:**
1. `runtimeMinutes` (Numeric — coalesced from IMDb + TMDB)
2. `numVotes` (Numeric)
3. `startYear` (Categorical)
4. `endYear` (Categorical)

**Network / Graph Centrality Features:**
5. `director_avg_centrality` (Numeric)
6. `director_count` (Numeric)
7. `writer_avg_centrality` (Numeric)
8. `writer_count` (Numeric)
9. `writer_writer_max_collab_weight` (Numeric)
10. `writer_director_max_collab_weight` (Numeric)

**TMDB External Features:**
11. `tmdb_popularity` (Numeric)
12. `tmdb_vote_average` (Numeric)
13. `tmdb_budget` (Numeric)
14. `tmdb_revenue` (Numeric)
15. `tmdb_primary_genre` (Fingerprint-keyed string → XGBoost native categorical)
16. `tmdb_original_language` (ISO 639-1 code → XGBoost native categorical)
17. `tmdb_origin_country` (ISO 3166-1 code → XGBoost native categorical)
18. `tmdb_production_company` (Fingerprint-keyed, frequency-thresholded → XGBoost native categorical)

**Engineered Interaction Features:**
19. `budget_revenue_ratio` (Numeric — budget/revenue, 0 when revenue is 0)
20. `votes_x_popularity` (Numeric — numVotes × tmdb_popularity)

> **Note:** `tmdb_success` and `synthetic_index` are metadata columns excluded from the feature matrix.

## Directory Structure
```text
big_data_project_2026/
├── requirements.txt             # Core dependencies (PySpark, XGBoost, Optuna, scikit-learn)
├── prepare_data.py              # Phase 1–3: Ingestion + Cleaning + Enrichment + Graph
├── analyze_covariance.py        # Spearman collinearity heatmap across the feature matrix
├── run_experiments.py           # Grid search over imputer configs + XGBoost Optuna tuning
├── inference.py                 # Unseen evaluation (test/validation prediction outputs)
├── src/
│   ├── config.py                # Central configuration and directory paths
│   ├── ingestion.py             # PySpark data cleaning + MAD outlier clamping
│   ├── tmdb_enrichment.py       # TMDB API fetch (concurrent) + categorical normalization
│   ├── graph_features.py        # Bipartite graph feature computation (repartitioned joins)
│   ├── imputation.py            # MLP-based multivariate imputation (optional)
│   ├── modeling.py              # XGBoost training & Optuna optimization
│   └── mlops.py                 # Drift detection (K-S, PSI, KL-Divergence)
├── data/
│   ├── train-*.csv              # Training CSVs (sharded)
│   ├── validation_hidden.csv    # Validation set
│   ├── test_hidden.csv          # Test set
│   ├── directing.json           # Director–movie relations
│   └── writing.json             # Writer–movie relations
└── output/
    ├── parquet/                  # Intermediate Parquet checkpoints
    ├── models/                  # XGBoost joblib models + vocab files + SUPREME_WINNER artifacts
    ├── analysis/                # Correlation heatmaps (from analyze_covariance.py)
    ├── experiment_results/      # ROC curves, confusion matrices, JSON stats, misclassification CSV
    ├── inference_parquet/       # Isolated inference-time Parquet outputs
    ├── logs/                    # TMDB API audit logs
    └── submissions/             # Final prediction outputs (*_predictions.txt)
```

## Getting Started

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Data Preparation (Run Once):**
    Executes PySpark ingestion, MAD clamping, concurrent TMDB API fetch, graph feature extraction, and categorical normalization.

    *Phase toggles:*
    - `--phase1`: Ingestion + MAD clamping only
    - `--phase2`: TMDB enrichment + categorical normalization only
    - `--phase3`: Graph feature computation only
    - *(No flags → runs all three phases sequentially)*

    ```bash
    python prepare_data.py
    ```

3.  **Evaluate Feature Collinearity (Optional):**
    Generates a Spearman rank correlation heatmap at `output/analysis/feature_correlation_heatmap.png`.

    *Command-line configurations:*
    - `--threshold <float>`: Absolute correlation threshold to flag as highly collinear (default: 0.75).

    ```bash
    python analyze_covariance.py
    ```

4.  **Run the Global Experimentation Suite:**
    Grid-searches over imputer hyperparameters. Tags the winning model explicitly. Generates an out-of-fold misclassification report.

    *Command-line configurations:*
    - `--enable-imputation`: Enable deep imputation (disabled by default).
    - `--massive`: Execute XGBoost optimization via distributed PySpark clusters.

    ```bash
    python run_experiments.py
    ```

5.  **Execute the Inference Predictor:**
    Automatically loads `SUPREME_WINNER_MODEL.joblib`. Outputs timestamped `True`/`False` text files to `output/submissions/`. Per-target TMDB caching prevents redundant API calls.

    *Command-line configurations:*
    - `--test_files <list>`: Space-separated list of CSV files to run inference on (default: `validation_hidden.csv test_hidden.csv`).
    - `--enable-imputation`: Enable deep imputation during inference if it was used in training.

    ```bash
    python inference.py
    ```
