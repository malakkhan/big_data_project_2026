"""
Central configuration for the IMDB Machine Learning Pipeline.

This module defines global constants, execution parameters, and directory
structures utilized across the PySpark ingestion, DuckDB feature extraction,
Deep Learning imputation, and XGBoost modeling phases.

Attributes:
    BASE_DIR (Path): The root directory of the project.
    DATA_DIR (Path): Directory containing raw input data.
    OUTPUT_DIR (Path): Directory for processed artifacts and models.
    TMDB_API_KEY (str): API key for The Movie Database.
    TMDB_READ_TOKEN (str): Bearer token for read-access to TMDB.
    SPARK_APP_NAME (str): Name of the PySpark application.
    SPARK_MASTER (str): The master URL for the Spark cluster.
    MAD_THRESHOLD_MULTIPLIER (float): Standard deviation multiplier for MAD outlier detection.
    IMPUTER_EPOCHS (int): Training epochs for the neural net imputer.
    IMPUTER_BATCH_SIZE (int): Batch size for imputer network training.
    IMPUTER_LEARNING_RATE (float): Multi-layer perceptron learning rate.
    XGB_PARAM_BOUNDS (dict): Hyperparameter search space for Optuna.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
PARQUET_DIR = OUTPUT_DIR / "parquet"

def create_directories():
    """
    Initializes required directory structures for output artifacts.

    Ensures that Parquet files, experiment results, and trained models
    have corresponding directories to persist into.

    Returns:
        None
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "models").mkdir(exist_ok=True)
    (OUTPUT_DIR / "experiment_results").mkdir(exist_ok=True)

# Call on import to guarantee availability
create_directories()

# TMDB API Configuration
TMDB_API_KEY = "a8e2a038b5579444c0dc4ac7d11c6a72"
TMDB_READ_TOKEN = (
    "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJhOGUyYTAzOGI1NTc5NDQ0YzBkYzRhYzdkMTFjNmE3MiIs"
    "Im5iZiI6MTc3NDAxMDY3My42MTEsInN1YiI6IjY5YmQ0MTMxMDM1YWZjMjg2YjQ1ZjQ2ZCIsInNj"
    "b3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.FSp0-R21yvAe3XjMxYAhEkiAUkTDgCf_Q"
    "vDnoiKdEz0"
)

# Programmatically broadcast hardcoded configs directly into global OS Environments
os.environ["TMDB_API_KEY"] = TMDB_API_KEY
os.environ["TMDB_READ_TOKEN"] = TMDB_READ_TOKEN

# PySpark Configuration
SPARK_APP_NAME = "IMDB_Pipeline"
SPARK_MASTER = "local[*]"

# Outlier Detection (MAD) Configurations (Mutable by global experiments)
MAD_THRESHOLD_MULTIPLIER = 4.0

# Imputer Modeling Configurations (Mutable by global experiments)
IMPUTER_EPOCHS = 5
IMPUTER_BATCH_SIZE = 128
IMPUTER_LEARNING_RATE = 0.001

# XGBoost Configuration Boundaries
XGB_PARAM_BOUNDS = {
    'max_depth': [3, 10],
    'learning_rate': [0.01, 0.3],
    'n_estimators': [50, 600],
    'reg_alpha': [0, 40], # L1 penalty
    'reg_lambda': [0, 40] # L2 penalty
}
