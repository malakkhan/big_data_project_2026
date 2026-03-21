"""
Inference & Prediction Pipeline.

Seamlessly loads structurally unseen data matrices through the entire
pipeline (Ingestion -> Graph Mapping -> DuckDB -> Imputer) and calculates
probabilities via the champion XGBoost Bayesian Optuna Model.
"""

import sys
import logging
import argparse
import shutil
import joblib
import pandas as pd
from pathlib import Path

from src import config

# Override Parquet directory globally for inference executions to prevent
# overwriting the foundational `featured_graph.parquet` training matrices!
config.PARQUET_DIR = config.OUTPUT_DIR / "inference_parquet"
config.PARQUET_DIR.mkdir(parents=True, exist_ok=True)

from src.ingestion import PySparkIngestor
from src.graph_features import GraphFeatureExtractor
from src.duckdb_processor import DuckDBFeatureEngineer
from src.imputation import DeepImputer

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("InferenceEngine")

def run_inference(targets, disable_imputation):
    import json
    winner_config_path = config.OUTPUT_DIR / "models" / "SUPREME_WINNER_CONFIG.json"
    if not winner_config_path.exists():
        logger.error("FATAL: SUPREME_WINNER_CONFIG.json not found! You must successfully complete `run_experiments.py` first.")
        sys.exit(1)
        
    with open(winner_config_path, "r") as f:
        winner_config = json.load(f)
        
    mad = winner_config.get("MAD", 3.0)
    epochs = winner_config.get("Epochs", 10)
    bs = winner_config.get("BatchSize", 128)
    lr = winner_config.get("LR", 0.001)
    
    for target in targets:
        logger.info(f"=========== PROCESSING EVALUATION SET: {target} ===========")
        
        macro_prefix = "SUPREME_WINNER"
        model_path = config.OUTPUT_DIR / "models" / "SUPREME_WINNER_MODEL.joblib"
        
        if not model_path.exists():
            logger.error(f"FATAL: Champion Model memory {model_path} not found! Have you executed `run_experiments`?")
            sys.exit(1)
            
        # Phase 1: Pure PySpark Ingestion (Dynamic Binding)
        logger.info(f"Phase 1: Passing Source '{target}' into Distributed PySpark Ingestion...")
        ingestor = PySparkIngestor()
        ingestor.run(target_pattern=target)
        
        # Phase 2: PySpark Topological Graphs and TMDB API Hooks
        logger.info(f"Phase 2: Hydrating '{target}' via Graph API and Collaborator Arrays...")
        graph_extractor = GraphFeatureExtractor()
        graph_extractor.run()
        
        # Phase 3: DuckDB Subsetting
        logger.info(f"Phase 3: Bypassing DuckDB Outlier Removal for Unseen Inference; Generating Latent Lexical TF-IDF matrices...")
        duckdb_processor = DuckDBFeatureEngineer()
        duckdb_processor.run(is_inference=True)
        
        # Phase 4: Imputation Restorations
        if disable_imputation:
            logger.info("Phase 4: Skipping Scikit-Learn Multilayer Perceptron Imputation Pipeline...")
            shutil.copy(config.PARQUET_DIR / "duckdb_features.parquet", config.PARQUET_DIR / "imputed_features.parquet")
        else:
            logger.info("Phase 4: Bridging Unseen Rows through Neural Vector Imputer...")
            config.IMPUTER_EPOCHS = epochs
            config.IMPUTER_BATCH_SIZE = bs
            config.IMPUTER_LEARNING_RATE = lr
            imputer = DeepImputer()
            imputer.run()
            
        # Phase 5: XGBoost Native Inferences
        logger.info(f"Phase 5: Loading XGBoost Optuna Artifact '{macro_prefix}' into Memory...")
        model = joblib.load(model_path)
        
        df = pd.read_parquet(config.PARQUET_DIR / "imputed_features.parquet")
        
        # Mirroring production alignments
        drop_cols = ["tconst", "primaryTitle", "originalTitle", "synthetic_index", "label"]
        feature_cols = [c for c in df.columns if c not in drop_cols]
        
        # Ensuring features passed match model strictly
        for expected_col in model.feature_names_in_:
            if expected_col not in feature_cols:
                df[expected_col] = 0 
                feature_cols.append(expected_col)
                
        X_test = df[model.feature_names_in_].copy()
        
        # Strictly map untamed test strings into exact Pandas categorical frameworks corresponding to XGBoost enable_categorical=True
        import json
        maps_path = config.OUTPUT_DIR / "models" / f"{macro_prefix}_categorical_maps.json"
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
            
        logger.info(f"Triggering massive probability regressions on '{target}'!")
        preds = model.predict_proba(X_test)[:, 1]
        
        # Format explicitly for submission constraints 
        import numpy as np
        
        # Output DataFrame with shuffled PySpark mapping
        output_df = pd.DataFrame({
            "tconst": df["tconst"],
            "predicted_label": (preds >= 0.5).astype(int)
        })
        
        # Graders strictly require identical row counts and identical sequence ordering!
        # PySpark hashing inherently shuffles rows and drops disjoint JSON metadata IDs.
        # We must Left-Join our XGBoost vectors directly back onto the chronological source file.
        original_csv = pd.read_csv(config.DATA_DIR / target)
        
        # Bind regressions sequentially
        aligned_df = pd.merge(original_csv[['tconst']], output_df, on='tconst', how='left')
        
        # If any rows were destroyed by PySpark/DuckDB parsing engines, default their blank inferences to False
        aligned_df['predicted_label'] = aligned_df['predicted_label'].fillna(0)
        
        # Convert binary ints to exactly "True" or "False" strings line by line
        boolean_predictions = np.where(aligned_df['predicted_label'] == 1, "True", "False")
        
        out_path = config.OUTPUT_DIR / f"{Path(target).stem}_predictions.txt"
        with open(out_path, "w") as f:
            for val in boolean_predictions:
                f.write(f"{val}\n")
                
        logger.info("======================================================")
        logger.info(f"✅ Inferences Computed for {target}! Predictions saved natively to: {out_path}")
        logger.info("======================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMDB Pipeline Unseen Inference Predictor")
    parser.add_argument("--test_files", nargs="+", default=["validation_hidden.csv", "test_hidden.csv"], help="List of file names bridging dynamic test predictions.")
    parser.add_argument("--disable-imputation", action="store_true", help="Bypass deep neural imputation logic")
    
    args = parser.parse_args()
            
    run_inference(args.test_files, args.disable_imputation)
