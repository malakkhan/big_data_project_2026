# big_data_project_2026
IMDB Big Data Project: Group 5
## Overview
This repository contains a production-grade machine learning pipeline designed to predict movie ratings using the IMDb dataset. The project integrates distributed data processing (PySpark), high-performance analytical querying (DuckDB), and advanced gradient boosting (XGBoost) within a modular, Object-Oriented architecture.

## Architecture & Key Features
The pipeline is divided into six decoupled phases, ensuring scalability and maintainability:

1.  **Distributed Ingestion:** Handles schema evolution, fixes temporal misalignments, and flattens nested JSON structures. (Responsible script: `src/ingestion.py`)
2.  **Graph Feature Engineering:** Constructs bipartite networks to calculate node centralities and collaborative weights. (Responsible script: `src/graph_features.py`)
3.  **Vectorized Processing:** Utilizes DuckDB SQL window functions for Robust Univariate Outlier Detection (MAD) and Hampel X84 filtration. (Responsible script: `src/duckdb_processor.py`)
4.  **Deep Imputation:** A DataWig-inspired module using character n-gram hashing and MLP networks to recover missing values. (Responsible script: `src/imputation.py`)
5.  **Bayesian Optimization:** Leverages Optuna to tune XGBoost hyperparameters, optimizing for ROC-AUC via Stratified K-Fold. (Responsible script: `src/modeling.py`)
6.  **MLOps & Drift Governance:** Implements K-S Test, PSI, and KL Divergence to detect covariate and semantic drift. (Responsible script: `src/mlops.py`)
7.  **Global Experimentation Engine:** In-memory grid-search evaluator looping through massive hyperparameters combinations. Generates explicit ROC curves, Confusion Matrices, and performance statistics securely across configurations. (Responsible script: `experiment_runner.py`)
8.  **Unseen Inference Engine:** End-to-end evaluation pipeline securely predicting unseen testing arrays without corrupting metrics via artifact deletions, exporting strict logical `True`/`False` text files natively. (Responsible script: `inference.py`)

## Directory Structure;  
```text
big_data_project_2026/
├── requirements.txt         # Core dependencies (PySpark, DuckDB, XGBoost, Optuna)
├── prepare_data.py          # Phase 1 & 2: PySpark Ingestion + Graph + TMDB API
├── run_experiments.py       # Phase 3-5: Grid Search over DuckDB, Imputation, & Models
├── inference.py             # Unseen evaluation scripting (test/validation outputs)
├── src/
│   ├── ingestion.py         # PySpark data cleaning & normalization
│   ├── graph_features.py    # Network analysis & bipartite graphs
│   ├── duckdb_processor.py  # Outlier detection & SVD dimensionality reduction
│   ├── imputation.py        # MLP-based multivariate imputation
│   ├── modeling.py          # XGBoost training & Optuna optimization
│   └── mlops.py             # Drift detection (K-S, PSI, KL-Divergence)
├── output/
│   ├── parquet/             # Intermediate processed Arrow states
│   ├── models/              # Finalized XGBoost joblib deployments
│   └── experiment_results/  # Generates `[trial]_roc.png`, `[trial]_cm.png`, and `[trial].json` statistics
├── config/                  # Configuration parameters and paths
└── walkthrough.md           # Detailed technical implementation guide
```

## Getting Started

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Data Preparation (Run Once):** 
    Executes heavy Spark JVM loads and REST API fetch mechanisms. Generates permanent structural datasets in `/output/parquet`
    
    *Command-line configurations:*
    - `--phase1`: Isolate execution to Phase 1 exclusively (PySpark Ingestion & JSON Flattening).
    - `--phase2`: Isolate execution to Phase 2 exclusively (Graph Extraction & TMDB API Hydration).
    - *(Passing no flags executes the entire architecture sequentially)*

    ```bash
    python prepare_data.py
    ```

3.  **Run the Global Experimentation Suite:** 
    Executes deep macroscopic hyperparameter grid searches decoupled from the intensive structural parses above. Tests multiple `duckdb` configurations and imputer schemas independently.
    
    *Command-line configurations:*
    - `--disable-imputation`: Bypass the DeepImputation module entirely, bridging DuckDB vectors linearly to XGBoost.

    ```bash
    python run_experiments.py
    ```
    *All evaluation matrices, ROC plots, and statistical JSON sheets will dynamically output into `/output/experiment_results/`!*

4.  **Execute the Inference Predictor:**
    Takes the precise macroscopic combination that "won" the experiments suite and pipes pure unseen CSV testing arrays into the generated Optuna `.joblib` model. Outputs strictly formatted `True`/`False` text files natively per grader constraints.

    *Command-line configurations:*
    - `--test_files` (str): Sequential list of target filenames to evaluate (defaults to `validation_hidden.csv test_hidden.csv`).
    - `--mad` (float): The final MAD threshold established by the winning configuration.
    - `--epochs` (int): Number of executing Imputer epochs initialized.
    - `--bs` (int): Scale of Imputer batch sizes.
    - `--lr` (float): Neural Learning Rate factor.
    - `--disable-imputation`: Boolean flag to skip neural imputation structures dynamically (must strictly replicate the winning configuration).

    ```bash
    python inference.py --mad 3.0 --epochs 10 --bs 128 --lr 0.001 --disable-imputation
    ```
