"""
Inference & Prediction Pipeline.

Loads structurally unseen data through the pipeline
(Ingestion -> TMDB Enrichment + Genre -> Graph Features -> Imputer) 
and predicts via the champion XGBoost model tagged as SUPREME_WINNER.

Design:
    A single SparkSession is created and shared across all pipeline phases
    to avoid redundant JVM overhead. Legacy Parquet format is enabled so
    Spark-written Parquet can be read by downstream pd.read_parquet().
"""

import sys
import logging
import argparse
import shutil
import json
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from pyspark.sql import SparkSession

from src import config

# Override Parquet directory globally for inference executions to prevent
# overwriting the foundational training matrices!
config.PARQUET_DIR = config.OUTPUT_DIR / "inference_parquet"
config.PARQUET_DIR.mkdir(parents=True, exist_ok=True)

from src.ingestion import PySparkIngestor
from src.tmdb_enrichment import TMDBEnrichment
from src.graph_features import GraphFeatureExtractor
from src.imputation import DeepImputer

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("InferenceEngine")

def _build_spark() -> SparkSession:
    """Shared SparkSession for inference with legacy Parquet format."""
    return SparkSession.builder \
        .appName(config.SPARK_APP_NAME) \
        .master(config.SPARK_MASTER) \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.ansi.enabled", "false") \
        .config("spark.sql.parquet.writeLegacyFormat", "true") \
        .getOrCreate()

def run_inference(targets, disable_imputation):
    winner_config_path = config.OUTPUT_DIR / "models" / "SUPREME_WINNER_CONFIG.json"
    if not winner_config_path.exists():
        logger.error("FATAL: SUPREME_WINNER_CONFIG.json not found! Run run_experiments.py first.")
        sys.exit(1)
        
    with open(winner_config_path, "r") as f:
        winner_config = json.load(f)
    
    macro = winner_config.get("macro_config", winner_config)
    epochs = macro.get("Epochs", 10)
    bs = macro.get("BatchSize", 128)
    lr = macro.get("LR", 0.001)
    
    model_path = config.OUTPUT_DIR / "models" / "SUPREME_WINNER_MODEL.joblib"
    if not model_path.exists():
        logger.error(f"FATAL: Champion Model {model_path} not found! Run run_experiments.py first.")
        sys.exit(1)
    
    # Single shared SparkSession for all phases
    spark = _build_spark()
    
    for target in targets:
        logger.info(f"=========== PROCESSING EVALUATION SET: {target} ===========")
        
        # Derive a label for per-target TMDB caching (e.g. "validation_hidden")
        tmdb_label = Path(target).stem  # "validation_hidden" or "test_hidden"

        # Phase 1: PySpark Ingestion (no MAD cleaning for inference)
        logger.info(f"Phase 1: PySpark Ingestion for '{target}'...")
        ingestor = PySparkIngestor(spark=spark)
        movies_spark_df = ingestor.run(target_pattern=target)
        
        # Write cleaned_data.parquet via Spark (legacy format, no .toPandas())
        cleaned_path = config.PARQUET_DIR / "cleaned_data.parquet"
        movies_spark_df.write.mode("overwrite").parquet(str(cleaned_path))
        
        # Phase 2: TMDB Enrichment + Genre Encoding (inference mode)
        # Each target set caches its TMDB data as tmdb_{label}.parquet
        logger.info(f"Phase 2: TMDB Enrichment + Genre Encoding for '{target}'...")
        enricher = TMDBEnrichment(spark=spark)
        enricher.run(is_inference=True, tmdb_label=tmdb_label)
        
        # Phase 3: Graph Features
        logger.info(f"Phase 3: Graph Feature Computation for '{target}'...")
        graph_extractor = GraphFeatureExtractor(spark=spark)
        graph_extractor.run()
        
        # Phase 4: Imputation
        if disable_imputation:
            logger.info("Phase 4: Skipping Imputation...")
            enriched_path = config.PARQUET_DIR / "enriched_features.parquet"
            imputed_path = config.PARQUET_DIR / "imputed_features.parquet"
            # Clean up any stale version (may be a file or Spark directory)
            if imputed_path.is_dir():
                shutil.rmtree(imputed_path)
            elif imputed_path.exists():
                imputed_path.unlink()
            shutil.copy(enriched_path, imputed_path)
        else:
            logger.info("Phase 4: Neural Imputation...")
            config.IMPUTER_EPOCHS = epochs
            config.IMPUTER_BATCH_SIZE = bs
            config.IMPUTER_LEARNING_RATE = lr
            imputer = DeepImputer()
            imputer.run()
            
        # Phase 5: XGBoost Prediction
        logger.info("Phase 5: Loading SUPREME_WINNER model...")
        model = joblib.load(model_path)
        
        df = pd.read_parquet(config.PARQUET_DIR / "imputed_features.parquet")
        
        drop_cols = ["tconst", "primaryTitle", "originalTitle", "synthetic_index", "label", "tmdb_success"]
        feature_cols = [c for c in df.columns if c not in drop_cols]
        
        for expected_col in model.feature_names_in_:
            if expected_col not in feature_cols:
                df[expected_col] = 0 
                feature_cols.append(expected_col)
                
        X_test = df[model.feature_names_in_].copy()
        
        maps_path = config.OUTPUT_DIR / "models" / "SUPREME_WINNER_MAPS.json"
        if maps_path.exists():
            with open(maps_path, 'r') as f:
                categorical_maps = json.load(f)
            for col in X_test.select_dtypes(include=['object']).columns:
                if col in categorical_maps:
                    X_test[col] = pd.Categorical(X_test[col].fillna("Unknown"), categories=categorical_maps[col])
                else:
                    X_test[col] = X_test[col].fillna("Unknown").astype('category')
        else:
            for col in X_test.select_dtypes(include=['object']).columns:
                X_test[col] = X_test[col].fillna("Unknown").astype('category')
            
        logger.info(f"Predicting on '{target}'...")
        preds = model.predict_proba(X_test)[:, 1]
        
        output_df = pd.DataFrame({
            "tconst": df["tconst"],
            "predicted_label": (preds >= 0.5).astype(int)
        })
        
        original_csv = pd.read_csv(config.DATA_DIR / target)
        aligned_df = pd.merge(original_csv[['tconst']], output_df, on='tconst', how='left')
        aligned_df['predicted_label'] = aligned_df['predicted_label'].fillna(0)
        
        boolean_predictions = np.where(aligned_df['predicted_label'] == 1, "True", "False")
        
        submissions_dir = config.OUTPUT_DIR / "submissions"
        submissions_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = submissions_dir / f"{Path(target).stem}_predictions_{timestamp}.txt"
        with open(out_path, "w") as f:
            for val in boolean_predictions:
                f.write(f"{val}\n")
                
        logger.info("=" * 60)
        logger.info(f"Inferences saved to: {out_path}")
        logger.info("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMDB Pipeline Inference Predictor")
    parser.add_argument("--test_files", nargs="+", default=["validation_hidden.csv", "test_hidden.csv"])
    parser.add_argument("--enable-imputation", action="store_true")
    
    args = parser.parse_args()
    run_inference(args.test_files, disable_imputation=not args.enable_imputation)
