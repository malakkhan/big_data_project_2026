"""
Global Experimentation Engine.

Iterates through macroscopic structural configurations (MAD multipliers and 
deep learning imputer dimensions) alongside micro-algorithmic XGBoost tuning.

Assumes that prepare_data.py has already been run and that
enriched_features.parquet is available in the output/parquet directory.
"""

import sys
import logging
import itertools
import shutil
import json
from pathlib import Path

from src import config
from src.imputation import DeepImputer
from src.modeling import XGBoostModeler

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("GlobalExperimentRunner")

def run_experiments(disable_imputation=False):
    """
    Executes a comprehensive grid search across macro pipeline parameters.
    
    Assumes enriched_features.parquet already exists from prepare_data.py.
    The experiment loop now only runs: optional imputation -> XGBoost.
    """
    config.create_directories()
    
    # Verify that enriched features exist
    enriched_path = config.PARQUET_DIR / "enriched_features.parquet"
    if not enriched_path.exists():
        logger.error("FATAL: enriched_features.parquet not found! Run prepare_data.py first.")
        sys.exit(1)
    
    # Define hyperparameter macro-grid
    imputer_epochs = [5, 10]
    imputer_batch_sizes = [64, 128]
    imputer_lrs = [1e-3, 5e-3]
    
    if disable_imputation:
        # Single pass — imputation params are irrelevant
        combinations = [(5, 64, 1e-3)]
    else:
        combinations = list(itertools.product(
            imputer_epochs, imputer_batch_sizes, imputer_lrs
        ))
    
    logger.info(f"Initialized Global Experiment Runner. Testing {len(combinations)} parameter topologies.")
    
    best_overall_auc = 0.0
    best_overall_config = None
    best_macro_prefix = None
    
    for idx, (epochs, bs, lr) in enumerate(combinations):
        logger.info("="*60)
        logger.info(f"RUNNING EXPERIMENT BLOCK {idx+1}/{len(combinations)}")
        logger.info(f"Parameters: Epochs={epochs}, BatchSize={bs}, LR={lr}")
        logger.info("="*60)
        
        # Inject macro-parameters into runtime config
        config.IMPUTER_EPOCHS = epochs
        config.IMPUTER_BATCH_SIZE = bs
        config.IMPUTER_LEARNING_RATE = lr
        
        macro_prefix = f"exp_ep{epochs}_bs{bs}_lr{lr}"
        
        # 1. Imputation Phase
        if disable_imputation:
            logger.info("-> Neural Imputation DISABLED. Bridging enriched features straight to XGBoost...")
            macro_prefix = "exp_no_imputation"
            enriched_src = config.PARQUET_DIR / "enriched_features.parquet"
            imputed_dst  = config.PARQUET_DIR / "imputed_features.parquet"
            # Clean up any stale version (may be a file or Spark directory)
            if imputed_dst.is_dir():
                shutil.rmtree(imputed_dst)
            elif imputed_dst.exists():
                imputed_dst.unlink()
            shutil.copy(enriched_src, imputed_dst)
        else:
            logger.info("-> Executing Neural Imputation")
            imputer = DeepImputer()
            imputer.run()
        
        # 2. XGBoost & Optuna Bayesian Phase
        logger.info("-> Executing XGBoost Search Space")
        modeler = XGBoostModeler(experiment_prefix=macro_prefix)
        current_auc = modeler.run()
        
        if current_auc is not None and current_auc > best_overall_auc:
            best_overall_auc = current_auc
            best_overall_config = {'Epochs': epochs, 'BatchSize': bs, 'LR': lr}
            best_macro_prefix = macro_prefix
            
    logger.info("="*60)
    logger.info("GLOBAL EXPERIMENTATION SUPREME CONFIGURATION")
    logger.info(f"Winning Configuration: {best_overall_config}")
    logger.info(f"Peak ROC-AUC Score: {best_overall_auc:.4f}")
    logger.info("="*60)
    
    winner_model = config.OUTPUT_DIR / "models" / f"{best_macro_prefix}_xgboost_best.joblib"
    winner_schema = config.OUTPUT_DIR / "models" / f"{best_macro_prefix}_feature_schema.json"
    winner_maps = config.OUTPUT_DIR / "models" / f"{best_macro_prefix}_categorical_maps.json"
    
    if winner_model.exists():
        shutil.copy(winner_model, config.OUTPUT_DIR / "models" / "SUPREME_WINNER_MODEL.joblib")
    if winner_schema.exists():
        shutil.copy(winner_schema, config.OUTPUT_DIR / "models" / "SUPREME_WINNER_SCHEMA.json")
    if winner_maps.exists():
        shutil.copy(winner_maps, config.OUTPUT_DIR / "models" / "SUPREME_WINNER_MAPS.json")
        
    # Load the winning model's trial JSON to capture Optuna-optimized params
    winner_trial_json = config.OUTPUT_DIR / "experiment_results" / f"{best_macro_prefix}_trial{0}.json"
    best_xgb_params = {}
    for trial_file in sorted((config.OUTPUT_DIR / "experiment_results").glob(f"{best_macro_prefix}_trial*.json")):
        # The last trial file from the best macro prefix has the best Optuna params
        with open(trial_file, "r") as f:
            trial_data = json.load(f)
            best_xgb_params = trial_data.get("hyperparameters", {})
    
    supreme_config = {
        "macro_config": best_overall_config,
        "best_roc_auc": round(best_overall_auc, 6),
        "optimized_xgboost_params": {
            "n_estimators": best_xgb_params.get("n_estimators"),
            "learning_rate": best_xgb_params.get("learning_rate"),
            "max_depth": best_xgb_params.get("max_depth"),
            "reg_alpha": best_xgb_params.get("reg_alpha"),
            "reg_lambda": best_xgb_params.get("reg_lambda"),
        }
    }
    
    with open(config.OUTPUT_DIR / "models" / "SUPREME_WINNER_CONFIG.json", "w") as f:
        json.dump(supreme_config, f, indent=4)
    
    logger.info("=" * 60)
    logger.info("OPTIMIZED XGBOOST HYPERPARAMETERS (Bayesian / Optuna):")
    logger.info(f"  n_estimators  = {supreme_config['optimized_xgboost_params']['n_estimators']}")
    logger.info(f"  learning_rate = {supreme_config['optimized_xgboost_params']['learning_rate']}")
    logger.info(f"  max_depth     = {supreme_config['optimized_xgboost_params']['max_depth']}")
    logger.info(f"  reg_alpha     = {supreme_config['optimized_xgboost_params']['reg_alpha']}")
    logger.info(f"  reg_lambda    = {supreme_config['optimized_xgboost_params']['reg_lambda']}")
    logger.info("=" * 60)
    
    # --- Misclassification analysis on training data ---
    logger.info("Generating misclassification report for the winning model...")
    try:
        import joblib
        import pandas as pd
        import numpy as np
        
        model = joblib.load(config.OUTPUT_DIR / "models" / "SUPREME_WINNER_MODEL.joblib")
        
        # Read the feature matrix (same one the model was trained on)
        input_parquet = config.PARQUET_DIR / "imputed_features.parquet"
        if not input_parquet.exists():
            input_parquet = config.PARQUET_DIR / "enriched_features.parquet"
        
        df = pd.read_parquet(input_parquet)
        
        id_cols = ["tconst", "primaryTitle", "originalTitle"]
        id_data = df[[c for c in id_cols if c in df.columns]].copy()
        
        drop_cols = ["tconst", "synthetic_index", "primaryTitle", "originalTitle", "C1", "tmdb_success"]
        feature_cols = [c for c in df.columns if c not in drop_cols and c != "label"]
        
        X = df[feature_cols].copy()
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].fillna("Unknown").astype('category')
        
        y_true = df["label"].astype(int)
        
        # Align features to model's expected schema
        for col in model.feature_names_in_:
            if col not in X.columns:
                X[col] = 0
        X = X[model.feature_names_in_]
        
        y_probs = model.predict_proba(X)[:, 1]
        y_pred = (y_probs >= 0.5).astype(int)
        
        misclassified_mask = y_pred != y_true
        n_wrong = misclassified_mask.sum()
        n_total = len(y_true)
        
        logger.info(f"Misclassified: {n_wrong}/{n_total} ({100 * n_wrong / n_total:.2f}%)")
        
        if n_wrong > 0:
            misclassified_df = id_data.loc[misclassified_mask].copy()
            misclassified_df["true_label"] = y_true.values[misclassified_mask]
            misclassified_df["predicted_prob"] = y_probs[misclassified_mask]
            misclassified_df["predicted_label"] = y_pred[misclassified_mask]
            
            # Include key features for error analysis
            key_features = [c for c in ["runtimeMinutes", "numVotes", "tmdb_popularity",
                                        "tmdb_vote_average", "tmdb_primary_genre",
                                        "tmdb_original_language", "tmdb_origin_country",
                                        "director_avg_centrality", "writer_avg_centrality"]
                           if c in df.columns]
            for col in key_features:
                misclassified_df[col] = df.loc[misclassified_mask, col].values
            
            out_csv = config.OUTPUT_DIR / "experiment_results" / "misclassified_examples.csv"
            misclassified_df.to_csv(out_csv, index=False)
            logger.info(f"Misclassified examples saved to {out_csv}")
    except Exception as e:
        logger.warning(f"Could not generate misclassification report: {e}")
        
    logger.info("All Global Experiments Concluded! Winning model tagged as 'SUPREME_WINNER_MODEL.joblib'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run IMDB Global Experiments")
    parser.add_argument("--disable-imputation", action="store_true", help="Skip the DeepImputation Module.")
    args = parser.parse_args()
    
    run_experiments(disable_imputation=args.disable_imputation)
